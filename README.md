# Bad Apple - Stable Diffusion Project

This project aims to recreate the iconic "Bad Apple!!" video using Stable Diffusion, utilizing a custom trained mapper model to guide the generation process based on the original video frames.

## Project Structure

- `src/`: Source code for the pipeline.
  - `extract_data.py`: Extracts frames and audio from the source video.
  - `generate_embeddings.py`: Generates CLIP embeddings for the frames.
  - `mapper_model.py`: Defines the neural network mapping model.
  - `train_mapper.py`: Training script for the mapper.
  - `inference.py`: Generation pipeline using the trained model.
  - `add_visuals.py`: Post-processing and video assembly.
- `data/`: Contains dataset and raw media.
- `models/`: Checkpoints and saved models.
- `output/`: Generated videos.

## Usage

1.  **Install dependencies:**
    ```bash
    uv sync
    ```

2.  **Run the pipeline:**
    ```bash
    python run.py
    ```

## Requirements
- Python 3.10+
- PyTorch with CUDA support
- Diffusers library
