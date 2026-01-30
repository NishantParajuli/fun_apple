
import os
import glob
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModelWithProjection, AutoProcessor, ClapModel
from tqdm import tqdm
import librosa

def generate_embeddings(dataset_dir="data/dataset", output_file="models/embeddings.pt", batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Load Models ---
    print("Loading CLIP (Image Encoder)...")
    # IP-Adapter SD1.5 uses ViT-H (1024 dim)
    clip_model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
    clip_model = CLIPVisionModelWithProjection.from_pretrained(clip_model_id).to(device)
    
    print("Loading CLAP (Audio Encoder)...")
    clap_model_id = "laion/clap-htsat-unfused"
    clap_processor = AutoProcessor.from_pretrained(clap_model_id)
    clap_model = ClapModel.from_pretrained(clap_model_id).to(device)

    # --- 2. Gather Files ---
    frame_files = sorted(glob.glob(os.path.join(dataset_dir, "frames", "*.jpg")))
    audio_files = sorted(glob.glob(os.path.join(dataset_dir, "audio", "*.wav")))
    
    if len(frame_files) == 0:
        raise ValueError("No data found! Run extract_data.py first.")
    if len(frame_files) != len(audio_files):
        print(f"Warning: Mismatch in file counts. Frames: {len(frame_files)}, Audio: {len(audio_files)}")
        # Truncate to min
        min_len = min(len(frame_files), len(audio_files))
        frame_files = frame_files[:min_len]
        audio_files = audio_files[:min_len]

    # --- 3. Processing Loop ---
    image_embeddings = []
    audio_embeddings = []

    print(f"Processing {len(frame_files)} pairs...")

    for i in tqdm(range(0, len(frame_files), batch_size)):
        batch_frames = frame_files[i : i + batch_size]
        batch_audio = audio_files[i : i + batch_size]
        
        # -- process images --
        images = [Image.open(f) for f in batch_frames]
        with torch.no_grad():
            inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(device)
            # visual_projection gives the implementation-generic "clip embedding" (768 dim for ViT-L/14)
            outputs = clip_model(**inputs)
            img_embeds = outputs.image_embeds # [B, 768]
            image_embeddings.append(img_embeds.cpu())

        # -- process audio --
        # CLAP requires raw audio arrays
        audios = []
        for af in batch_audio:
            y, sr = librosa.load(af, sr=48000) # Ensure consistent SR required by CLAP usually 48k
            audios.append(y)
            
        with torch.no_grad():
            # pad/truncate is handled by processor usually, but check params if needed.
            # default max_length often 10s, we have 1s clips, so padding is important.
            inputs = clap_processor(audio=audios, return_tensors="pt", sampling_rate=48000, padding=True).to(device)
            outputs = clap_model.get_audio_features(**inputs)
            if isinstance(outputs, torch.Tensor):
                aud_embeds = outputs
            else:
                print(f"DEBUG TYPE: {type(outputs)}")
                # Try common keys
                if hasattr(outputs, "pooler_output"):
                    aud_embeds = outputs.pooler_output
                elif hasattr(outputs, "last_hidden_state"):
                     # This might not be projected!
                     aud_embeds = outputs.last_hidden_state[:, 0, :]
                else:
                    # Fallback to tuple
                    aud_embeds = outputs[0]
            
            audio_embeddings.append(aud_embeds.cpu())

    # --- 4. Save ---
    all_img_emb = torch.cat(image_embeddings, dim=0)
    all_aud_emb = torch.cat(audio_embeddings, dim=0)
    
    print(f"Saving embeddings: Audio {all_aud_emb.shape}, Image {all_img_emb.shape}")
    torch.save({
        "audio_embeddings": all_aud_emb,
        "image_embeddings": all_img_emb,
        "filenames": frame_files # good for debugging/validation
    }, output_file)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    generate_embeddings()
