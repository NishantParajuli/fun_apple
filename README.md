# Bad Apple - Neural Audio-Visual Synchronization

> **"The Blind Painter" Experiment**  
> *Generating "Bad Apple!!" visuals purely from audio using Cross-Modal Latent Mapping.*

---

## 🎵 How It Works

This project explores **Cross-Modal Latent Mapping**. Instead of traditional audio-reactive visualization (which uses spectrum analysis to drive parameters), this system **"hears" the audio and hallucinates the corresponding visual frame**.

The pipeline bridges the gap between Audio encodings and Image encodings:

1.  **Audio Encoding (CLAP)**: The audio track is sliced into segments and fed into a **CLAP (Contrastive Language-Audio Pretraining)** model. This converts audio into a mathematical vector representing its semantic content.
2.  **The "Blind Painter" Mapper**: A custom-trained neural network (MLP) translates this **CLAP Audio Embedding** into a predicted **CLIP Image Embedding**. It learns the relationship between the song's audio features and the original video's visual composition.
3.  **Generative Reconstruction (SD 1.5 + IP-Adapter)**: The predicted image embedding is injected into **Stable Diffusion 1.5** using **IP-Adapter**. The model generates a new frame that "locks on" to the predicted concept.
4.  **Temporal Logic**: To prevent flickering, a custom algorithm manages:
    *   **Scene Polarity**: Detects light/dark scene switches in the original flow.
    *   **Onset Reactivity**: Hard cuts on strong beats, smoothing during sustained notes.
    *   **Temporal Smoothing**: Blends embeddings over time for fluid motion.

---

## 📂 Project Structure

- **`src/`**: Core implementation files.
    - **`extract_data.py`**: Preprocessing script. Extracts individual frames and the audio track from the source video.
    - **`generate_embeddings.py`**: Creates the training dataset. Runs frames through CLIP and audio segments through CLAP to create paired embeddings.
    - **`mapper_model.py`**: Defines the `BlindPainterMapper`, a lightweight MLP neural network.
    - **`train_mapper.py`**: Trains the Mapper to predict visual embeddings from audio embeddings.
    - **`inference.py`**: The main generation pipeline. Loads the trained mapper and generates the video frame-by-frame.
    - **`add_visuals.py`**: Post-processing script to overlay waveforms or concatenate audio.
- **`run.py`**: Central CLI entry point for all pipeline stages.
- **`data/`**: Directory for input media (source MP4) and extracted components.
- **`models/`**: Stores the generated `.pt` embedding datasets and trained `.pth` models.
- **`output/`**: Destination for generated video files.

---

## 🚀 Usage Guide

This project uses `run.py` as a unified commander for the pipeline.

### 0. Prerequisites
- **Python 3.10+**
- **FFmpeg** (Required for audio extraction and video assembly)
    ```bash
    sudo apt install ffmpeg
    ```
- **uv** (Recommended package manager)
    ```bash
    uv sync
    ```

### 1. Data Parsing (`extract`)
Prepare your source video (e.g., `data/bad_apple.mp4`). This step extracts raw frames and the audio track.
```bash
python run.py extract --video_path data/bad_apple.mp4
```

### 2. Generate Training Data (`generate`)
Compute embeddings for every frame (CLIP) and its corresponding audio segment (CLAP). This creates `models/embeddings.pt`.
```bash
python run.py generate
```

### 3. Train the Mapper (`train`)
Train the neural network to translate Audio -> Video.
```bash
python run.py train
```
*The best model will be saved to `models/mapper_best.pth`.*

### 4. Run Inference (`inference`)
Generate a new video from pure audio using your trained mapper.
```bash
# Basic usage
python run.py inference --audio_path data/bad_apple.mp4

# Quick test (first 300 frames)
python run.py inference --max_frames 300
```

### 5. Final Polish (`visuals`)
Combine visuals with the audio track and apply any post-processing overlays.
```bash
python run.py visuals
```

---

## 🛠️ Configuration
Most scripts accept arguments. use `--help` to see them.
```bash
python run.py inference --help
```

## 📦 Requirements
- `torch`, `torchvision`, `torchaudio` (CUDA highly recommended)
- `diffusers`, `transformers`
- `librosa`
- `opencv-python`
- `soundfile`
