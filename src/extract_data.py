
import os
import glob
import subprocess
import librosa
import soundfile as sf
import argparse
import numpy as np
from tqdm import tqdm
import shutil

def extract_data(video_path, output_dir="data/dataset", fps=10, audio_duration=1.0):
    """
    Extracts frames and audio using ffmpeg for robustness.
    """
    # Setup directories
    frames_base_dir = os.path.join(output_dir, "frames")
    audio_dir = os.path.join(output_dir, "audio")
    temp_frames_dir = os.path.join(output_dir, "temp_frames")
    
    if os.path.exists(temp_frames_dir):
        shutil.rmtree(temp_frames_dir)
    os.makedirs(temp_frames_dir, exist_ok=True)
    os.makedirs(frames_base_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    print(f"Processing {video_path}...")

    # 1. Extract all frames to temp dir using ffmpeg
    print("Extracting frames with ffmpeg...")
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        os.path.join(temp_frames_dir, "frame_%04d.jpg")
    ]
    subprocess.run(cmd, check=True)
    
    # 2. Extract full audio
    print("Loading audio...")
    y, sr = librosa.load(video_path, sr=48000)
    audio_len_samples = len(y)
    
    # 3. Match audio to frames
    frame_files = sorted(glob.glob(os.path.join(temp_frames_dir, "*.jpg")))
    print(f"Extracted {len(frame_files)} frames. syncing audio...")
    
    total_frames = len(frame_files)
    video_fps = fps # Since we forced fps in ffmpeg
    
    for i, frame_path in enumerate(tqdm(frame_files)):
        # Verify frame valid just in case
        if not os.path.getsize(frame_path):
            continue
            
        current_time = i / video_fps
        
        # Audio Window
        start_time = current_time - (audio_duration / 2)
        end_time = current_time + (audio_duration / 2)
        
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Padding
        if start_sample < 0:
            audio_segment = np.pad(y[0:end_sample], (abs(start_sample), 0), mode='constant')
        elif end_sample > audio_len_samples:
             audio_segment = np.pad(y[start_sample:], (0, end_sample - audio_len_samples), mode='constant')
        else:
            audio_segment = y[start_sample:end_sample]
            
        target_len = int(audio_duration * sr)
        if len(audio_segment) != target_len:
            if len(audio_segment) < target_len:
                 audio_segment = np.pad(audio_segment, (0, target_len - len(audio_segment)), mode='constant')
            else:
                 audio_segment = audio_segment[:target_len]

        # Move frame to final dir
        final_frame_name = f"frame_{i:04d}.jpg"
        shutil.move(frame_path, os.path.join(frames_base_dir, final_frame_name))
        
        # Save Audio
        audio_filename = f"frame_{i:04d}.wav"
        sf.write(os.path.join(audio_dir, audio_filename), audio_segment, sr)

    # Cleanup
    shutil.rmtree(temp_frames_dir)
    print(f"Done! Processed {len(frame_files)} pairs to '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="data/bad_apple.mp4", help="Path to Bad Apple video file")
    parser.add_argument("--fps", type=int, default=10, help="Sampling FPS")
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found. Please place 'bad_apple.mp4' in 'data/' directory or specify --video_path.")
    else:
        extract_data(args.video_path, fps=args.fps)
