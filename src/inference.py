
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image
from transformers import AutoProcessor, ClapModel
import librosa
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
from .mapper_model import BlindPainterMapper
import soundfile as sf

def inference(audio_path, mapper_path="models/mapper_best.pth", output_video="output/output.mp4", max_frames=None):
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    if os.path.dirname(mapper_path):
        os.makedirs(os.path.dirname(mapper_path), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    mapper = BlindPainterMapper().to(device)
    mapper.load_state_dict(torch.load(mapper_path, map_location=device))
    mapper.eval()
    
    clap_model_id = "laion/clap-htsat-unfused"
    clap_processor = AutoProcessor.from_pretrained(clap_model_id)
    clap_model = ClapModel.from_pretrained(clap_model_id).to(device) 
    
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16, 
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False
    )
    
    from diffusers import DPMSolverMultistepScheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    pipe.to(device)
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xFormers enabled")
    except Exception as e:
        print(e)
    
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    pipe.set_progress_bar_config(disable=True)

    print(f"Audio {audio_path}")
    y, sr = librosa.load(audio_path, sr=48000)
    duration = len(y) / sr
    fps = 10 
    total_frames = int(duration * fps)
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
        print(f"Limited to {total_frames} frames.")
    
    audio_clip_duration = 1.0 # 
    
    print(f"Generating {total_frames} frames")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter("output/temp_video.mp4", fourcc, fps, (512, 512))
    
    prev_embedding = None
    smoothing_factor = 0.6 
    
    prev_image = np.zeros((512, 512, 3), dtype=np.uint8)
    prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2RGB)
    from PIL import Image
    prev_image_pil = Image.fromarray(prev_image)
    
    scene_polarity = "dark"  # first frame is always black
    polarity_threshold_dark = 80   
    polarity_threshold_light = 175 
    
    for i in tqdm(range(total_frames)):
        current_time = i / fps
        
        start_sample = int((current_time - audio_clip_duration/2) * sr)
        end_sample = int((current_time + audio_clip_duration/2) * sr)
        
        if start_sample < 0:
            audio_segment = np.pad(y[0:end_sample], (abs(start_sample), 0), mode='constant')
        elif end_sample > len(y):
             audio_segment = np.pad(y[start_sample:], (0, end_sample - len(y)), mode='constant')
        else:
            audio_segment = y[start_sample:end_sample]
            
        inputs = clap_processor(audio=audio_segment, return_tensors="pt", sampling_rate=48000, padding=True).to(device)
        with torch.inference_mode():
            outputs = clap_model.get_audio_features(**inputs)
            if isinstance(outputs, torch.Tensor):
                aud_emb = outputs
            elif hasattr(outputs, "pooler_output"):
                aud_emb = outputs.pooler_output
            else:
                aud_emb = outputs[0]
                
        with torch.inference_mode():
            pred_img_emb = mapper(aud_emb) # [1, 1024]
            
        onset_env = librosa.onset.onset_strength(y=audio_segment, sr=sr)
        onset_score = np.mean(onset_env) if len(onset_env) > 0 else 0.0
        
        reactivity = np.clip(onset_score / 1.5, 0.0, 1.0) 
        
        if reactivity > 0.8:
            dyn_strength = 1.0 
            dyn_smoothing = 0.1 
        else:
            dyn_strength = 0.55 + (0.40 * reactivity)
            dyn_smoothing = 0.75 - (0.6 * reactivity)
        
        if prev_embedding is not None:
            cos_sim = F.cosine_similarity(pred_img_emb, prev_embedding, dim=-1).item()
            if cos_sim < 0.7:  
                dyn_strength = 1.0 
                dyn_smoothing = 0.0  
            pred_img_emb = (1 - dyn_smoothing) * pred_img_emb + dyn_smoothing * prev_embedding
        prev_embedding = pred_img_emb.clone()  
        
        ip_embeds = pred_img_emb.unsqueeze(1) 
        neg_ip_embeds = torch.zeros_like(ip_embeds)
        ip_embeds = torch.cat([neg_ip_embeds, ip_embeds], dim=0) 
        
        with torch.inference_mode():
            image = pipe(
                prompt="black and white silhouette, bad apple style, anime, high contrast",
                negative_prompt="color, realistic, photo, noise, blur",
                image=prev_image_pil,
                strength=dyn_strength, 
                ip_adapter_image_embeds=[ip_embeds], 
                num_inference_steps=15,  
                guidance_scale=7.5
            ).images[0]
            
        prev_image_pil = image
            
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        video_writer.write(img_bgr)
        
        frame_luminance = np.mean(img_np)
        new_polarity = scene_polarity
        if scene_polarity == "dark" and frame_luminance > polarity_threshold_light:
            new_polarity = "light"
        elif scene_polarity == "light" and frame_luminance < polarity_threshold_dark:
            new_polarity = "dark"
        
        if new_polarity != scene_polarity:
            if new_polarity == "dark":
                prev_image_pil = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
            else:
                prev_image_pil = Image.fromarray(np.full((512, 512, 3), 255, dtype=np.uint8))
            scene_polarity = new_polarity
        
    video_writer.release()
    
    os.system(f"ffmpeg -y -i output/temp_video.mp4 -i {audio_path} -c:v copy -c:a aac -shortest {output_video}")
    print(f"{output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default="data/bad_apple.mp4")
    parser.add_argument("--mapper_path", type=str, default="models/mapper_best.pth")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit number of frames for testing")
    args = parser.parse_args()
    
    inference(args.audio_path, args.mapper_path, max_frames=args.max_frames)
