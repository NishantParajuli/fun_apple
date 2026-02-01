
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # --- 1. Load Models ---
    print("Loading Mapper...")
    mapper = BlindPainterMapper().to(device)
    mapper.load_state_dict(torch.load(mapper_path, map_location=device))
    mapper.eval()
    
    print("Loading CLAP...")
    clap_model_id = "laion/clap-htsat-unfused"
    clap_processor = AutoProcessor.from_pretrained(clap_model_id)
    clap_model = ClapModel.from_pretrained(clap_model_id).to(device)  # Keep FP32 - BatchNorm doesn't support FP16
    
    print("Loading SD 1.5 + IP-Adapter...")
    
    print("Switching to SD 1.5 Img2Img for temporal stability...")
    # Disable Safety Checker to prevent false positives on abstract/silhouette art
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16, 
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Use DPMSolver for fast generation
    from diffusers import DPMSolverMultistepScheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    pipe.to(device)
    
    # Speed Optimizations
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xFormers enabled!")
    except Exception as e:
        print(f"xFormers not available: {e}")
    
    # torch.compile for major speedup (first run will be slower due to JIT)
    print("Compiling UNet with torch.compile...")
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    # Disable internal diffusers progress bars (we use our own tqdm)
    pipe.set_progress_bar_config(disable=True)

    # --- 2. Load Audio ---
    print(f"Loading Audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=48000)
    duration = len(y) / sr
    fps = 10 # Generation FPS
    total_frames = int(duration * fps)
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
        print(f"Limited to {total_frames} frames.")
    
    audio_clip_duration = 1.0 # The model was trained on 1s clips
    
    print(f"Generating {total_frames} frames...")
    
    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # SD1.5 generates 512x512 by default
    video_writer = cv2.VideoWriter("output/temp_video.mp4", fourcc, fps, (512, 512))
    
    # Smoothing buffer
    prev_embedding = None
    smoothing_factor = 0.6 # 0.0 = no smoothing, 1.0 = no change
    
    # Init previous image (Start with a blank or black image)
    # Bad apple starts black.
    prev_image = np.zeros((512, 512, 3), dtype=np.uint8)
    prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2RGB)
    from PIL import Image
    prev_image_pil = Image.fromarray(prev_image)
    
    # Luminance polarity tracking (Fix #2)
    scene_polarity = "dark"  # Bad Apple opens with dark background
    polarity_threshold_dark = 80   # Below this = dark scene
    polarity_threshold_light = 175  # Above this = light scene
    
    for i in tqdm(range(total_frames)):
        current_time = i / fps
        
        # ... (Audio Extraction - Skipping lines for brevity) ... 
        # Re-implementing the loop content to ensure context is right
        
        # Extract Audio Window
        start_sample = int((current_time - audio_clip_duration/2) * sr)
        end_sample = int((current_time + audio_clip_duration/2) * sr)
        
        if start_sample < 0:
            audio_segment = np.pad(y[0:end_sample], (abs(start_sample), 0), mode='constant')
        elif end_sample > len(y):
             audio_segment = np.pad(y[start_sample:], (0, end_sample - len(y)), mode='constant')
        else:
            audio_segment = y[start_sample:end_sample]
            
        # 1. Get Audio Embedding
        inputs = clap_processor(audio=audio_segment, return_tensors="pt", sampling_rate=48000, padding=True).to(device)
        with torch.inference_mode():
            outputs = clap_model.get_audio_features(**inputs)
            if isinstance(outputs, torch.Tensor):
                aud_emb = outputs
            elif hasattr(outputs, "pooler_output"):
                aud_emb = outputs.pooler_output
            else:
                aud_emb = outputs[0]
                
        # 2. Map to Image Embedding
        with torch.inference_mode():
            pred_img_emb = mapper(aud_emb) # [1, 1024]
            
        # 3. Dynamic Reactivity (The "Bad Apple" Fix)
        # Calculate Audio Onset Strength for this frame
        onset_env = librosa.onset.onset_strength(y=audio_segment, sr=sr)
        onset_score = np.mean(onset_env) if len(onset_env) > 0 else 0.0
        
        # Normalize roughly (observed values usually 0-10)
        # We want:
        # High Energy (Score > 1.0) -> High Change (Strength ~0.85), Low Smoothing (0.2)
        # Low Energy (Score < 0.2)  -> Low Change (Strength ~0.45), High Smoothing (0.8)
        
        # Sigmoid-like mapping (Aggressive)
        reactivity = np.clip(onset_score / 1.5, 0.0, 1.0) 
        
        # Hard Cut Logic for strong beats
        if reactivity > 0.8:
            dyn_strength = 1.0 # New Scene!
            dyn_smoothing = 0.1 # Mostly new embedding
        else:
            dyn_strength = 0.55 + (0.40 * reactivity) # 0.55 to 0.95
            dyn_smoothing = 0.75 - (0.6 * reactivity) # 0.75 to 0.15
        
        # Temporal Smoothing (Embedding Level)
        # Fix #3: Embedding-based scene detection
        if prev_embedding is not None:
            cos_sim = F.cosine_similarity(pred_img_emb, prev_embedding, dim=-1).item()
            if cos_sim < 0.7:  # Major scene change detected!
                dyn_strength = 1.0  # Full regeneration
                dyn_smoothing = 0.0  # No blending with old embedding
            pred_img_emb = (1 - dyn_smoothing) * pred_img_emb + dyn_smoothing * prev_embedding
        prev_embedding = pred_img_emb.clone()  # Clone to avoid reference issues
        
        # IP-Adapter shape fix
        ip_embeds = pred_img_emb.unsqueeze(1) 
        neg_ip_embeds = torch.zeros_like(ip_embeds)
        ip_embeds = torch.cat([neg_ip_embeds, ip_embeds], dim=0) 
        
        # 4. Generate Image (Img2Img)
        with torch.inference_mode():
            image = pipe(
                prompt="black and white silhouette, bad apple style, anime, high contrast",
                negative_prompt="color, realistic, photo, noise, blur",
                image=prev_image_pil,
                strength=dyn_strength, 
                ip_adapter_image_embeds=[ip_embeds], 
                num_inference_steps=15,  # Reduced from 20 - DPMSolver handles this well
                guidance_scale=7.5
            ).images[0]
            
        # Update prev frame
        prev_image_pil = image
            
        # Convert to CV2
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        video_writer.write(img_bgr)
        
        # Fix #2: Luminance polarity tracking
        frame_luminance = np.mean(img_np)
        new_polarity = scene_polarity
        if scene_polarity == "dark" and frame_luminance > polarity_threshold_light:
            new_polarity = "light"
        elif scene_polarity == "light" and frame_luminance < polarity_threshold_dark:
            new_polarity = "dark"
        
        if new_polarity != scene_polarity:
            # Polarity flipped! Reset prev_image to help SD adapt
            if new_polarity == "dark":
                prev_image_pil = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
            else:
                prev_image_pil = Image.fromarray(np.full((512, 512, 3), 255, dtype=np.uint8))
            scene_polarity = new_polarity
        
    video_writer.release()
    
    # Merge Audio
    print("Merging Audio...")
    os.system(f"ffmpeg -y -i output/temp_video.mp4 -i {audio_path} -c:v copy -c:a aac -shortest {output_video}")
    print(f"Done! Saved to {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default="data/bad_apple.mp4")
    parser.add_argument("--mapper_path", type=str, default="models/mapper_best.pth")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit number of frames for testing")
    args = parser.parse_args()
    
    inference(args.audio_path, args.mapper_path, max_frames=args.max_frames)
