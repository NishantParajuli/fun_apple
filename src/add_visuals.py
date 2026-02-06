
import os
import argparse
import subprocess

def add_visuals(input_video, output_video):
    """
    Adds a waveform overlay to the video using ffmpeg.
    """
    print(f"Processing {input_video} -> {output_video}")
    

    # Get duration of video stream
    cmd_dur = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=duration", "-of", "default=noprint_wrappers=1:nokey=1",
        input_video
    ]
    result = subprocess.run(cmd_dur, capture_output=True, text=True)
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        print("Warning: Could not determine video duration, fallback to copy full audio.")
        duration = None

    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-filter_complex", 
        "[0:a]showwaves=s=1280x240:mode=line:colors=cyan@0.5[sw];[0:v][sw]overlay=0:H-h:format=auto,format=yuv420p[v]",
        "-map", "[v]",
        "-map", "0:a",
        "-c:v", "libx264",
        "-c:a", "copy"
    ]
    
    if duration:
        cmd.extend(["-t", str(duration)])
        
    cmd.append(output_video)
    
    subprocess.run(cmd, check=True)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="output/output.mp4")
    parser.add_argument("--output", type=str, default="output/final_polished.mp4")
    args = parser.parse_args()
    
    if os.path.exists(args.input):
        add_visuals(args.input, args.output)
    else:
        print(f"Input file {args.input} not found.")
