import sys
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [inference|train|generate|extract|visuals] [args]")
        print("\nAvailable modes:")
        print("  inference  : Run the Stable Diffusion inference")
        print("  train      : Train the mapper model")
        print("  generate   : Generate embeddings from dataset")
        print("  extract    : Extract frames and audio from video")
        print("  visuals    : Add waveform visuals to output")
        return

    mode = sys.argv[1]
    args = sys.argv[2:]
    
    module_map = {
        "inference": "src.inference",
        "train": "src.train_mapper",
        "generate": "src.generate_embeddings",
        "extract": "src.extract_data",
        "visuals": "src.add_visuals"
    }
    
    if mode not in module_map:
        print(f"Unknown mode: {mode}. Available: {list(module_map.keys())}")
        return

    cmd = [sys.executable, "-m", module_map[mode]] + args
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        sys.exit(1)

if __name__ == "__main__":
    main()
