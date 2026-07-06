# AI Scouting / Robot Scouter

Robot Scouter is a local Gradio application for analyzing match video and producing robot scouting data. It can split a 720p or 1080p composite match recording into blue, center, and red camera views, calibrate the field, track robot movement, detect yellow fuel, read center-screen score/clock information with OCR, and generate annotated videos, map time-lapses, and per-robot scoring tables.

The app is designed to run locally on Windows or Linux. A CUDA-capable NVIDIA GPU is strongly recommended when using the SAM 3 fuel detector, but the app can fall back to HSV fuel detection when SAM is unavailable.

## Features

- Gradio web UI served from `app.py`
- Composite match video upload or YouTube download through `yt-dlp`
- Automatic, manual, manual-limited, and OCR-only workflows
- Center and side-camera calibration with saved regional calibration cache
- Robot movement maps and autonomous movement summaries
- Yellow fuel detection using SAM 3 or HSV
- Optional YOLO person segmentation to avoid counting humans as robots
- Optional Tesseract OCR for center score and match clock correction
- Optional local vision LLM support for robot/team-number labeling

## Repository Layout

| Path | Purpose |
| --- | --- |
| `app.py` | Main Gradio application |
| `requirements.txt` | Python package dependencies |
| `config.json` | Local app mode and LLM endpoint settings |
| `field_calibration_cache.json` | Saved calibration data by regional/event name |
| `reference_image.png`, `map.png` | Reference/map assets used by the app |
| `yolo26s-seg.pt` | YOLO segmentation weights for person detection |
| `sam3.1_multiplex.pt` | SAM 3 weights for fuel detection |
| `testgradio.py` | Example remote Gradio client script |

## Requirements

- Python 3.10 or newer. Python 3.11 is a good default.
- Windows 10/11 or a modern Linux distribution.
- Enough disk space for model weights. `sam3.1_multiplex.pt` is several GB.
- NVIDIA GPU and current NVIDIA driver for practical SAM 3 performance.
- Tesseract OCR if you want score/clock OCR correction.
- FFmpeg for video splitting and encoding. The `static-ffmpeg` Python package is included, but a system FFmpeg install is still useful if your platform has trouble with the bundled binary.

## Model Files

The app looks for these files in the repository root:

```text
yolo26s-seg.pt
sam3.1_multiplex.pt
```

If `yolo26s-seg.pt` is missing, person detection is disabled. If `sam3.1_multiplex.pt` is missing, the installed `ultralytics` version does not support SAM 3, or CUDA/Torch is not working, the app falls back to HSV fuel detection.

The SAM 3.1 checkpoint is not included in this GitHub repository. Download it from Meta's Hugging Face repository:

```text
https://huggingface.co/facebook/sam3.1
```

That model repository is gated. You will need to create or log in to a Hugging Face account, review and accept the license/access terms, and agree to share the required contact information before the checkpoint files are available.

After downloading the multiplex checkpoint, place it directly in this project folder and make sure the filename matches what `app.py` expects:

```text
ai-scouting/
  app.py
  sam3.1_multiplex.pt
  yolo26s-seg.pt
```

On Windows, this can be as simple as dragging `sam3.1_multiplex.pt` into the project folder in File Explorer. On Linux, copy or move it into the repository root:

```bash
cp /path/to/sam3.1_multiplex.pt .
```

## Install on Windows

Open PowerShell in the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

For CPU-only use or HSV fuel detection:

```powershell
pip install -r requirements.txt
```

For SAM 3 with an NVIDIA GPU, install a CUDA-enabled PyTorch build before installing the rest of the requirements. Use the official PyTorch selector if the command below no longer matches your driver/CUDA environment:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Install Tesseract OCR if you want OCR-based score and clock correction:

```powershell
winget install UB-Mannheim.TesseractOCR
```

The app checks the standard Windows Tesseract install locations automatically.

## Install on Linux

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For CPU-only use or HSV fuel detection:

```bash
pip install -r requirements.txt
```

For SAM 3 with an NVIDIA GPU, first verify the driver:

```bash
nvidia-smi
```

Then install a CUDA-enabled PyTorch build before the rest of the requirements. Use the official PyTorch selector if you need a different CUDA wheel:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Install optional system tools:

```bash
sudo apt update
sudo apt install -y tesseract-ocr ffmpeg
```

For non-Debian distributions, install the equivalent `tesseract` and `ffmpeg` packages through your package manager.

## Verify CUDA

After installation, confirm PyTorch can see your GPU:

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If this prints `CUDA available: False`, SAM 3 will usually be too slow or may fail depending on the installed Torch build. Reinstall PyTorch with the correct CUDA wheel for your system.

## Configuration

Edit `config.json` before starting the app:

```json
{
  "local_llm_url": "http://127.0.0.1:1234/v1/chat/completions",
  "robot_tracking_mode": "manual"
}
```

Supported `robot_tracking_mode` values:

| Mode | Behavior |
| --- | --- |
| `auto` | Full automatic workflow with composite video splitting, multi-camera processing, robot detection, fuel detection, OCR, and optional LLM labeling |
| `manual` | Center-camera workflow where you manually track robots, then let the app compute fuel/shot stats |
| `manual-limited` | Manual center-camera workflow that skips non-table outputs for faster processing |
| `ocr` | Center-camera workflow where you mark which robots are shooting and OCR score changes are split across marked robots |

## Run the App

Start the Gradio server:

```bash
python app.py
```

Open:

```text
http://localhost:7860
```

The app binds to `0.0.0.0:7860`, so on another device on the same network you can open:

```text
http://<host-ip>:7860
```

Stop the server with `Ctrl+C`.

## Basic Workflow

1. Choose the desired `robot_tracking_mode` in `config.json`.
2. Start `python app.py`.
3. Upload a composite match video or paste a YouTube match URL.
4. Enter the event/regional name if you want calibration saved and reused.
5. Enter the three blue and three red robot team numbers.
6. Calibrate the center camera points when prompted.
7. Choose detector options:
   - `SAM 3` for better fuel segmentation when CUDA is working.
   - `HSV` for a lighter, faster fallback.
8. Process the video and review the annotated videos, map output, and robot tables.

## Local Vision LLM Setup

The app can query a local OpenAI-compatible chat-completions endpoint for robot/team-number labeling. It sends cropped robot images, so use a vision-capable model, not a text-only model.

### Recommended: LM Studio

LM Studio is the easiest option because it gives you a desktop UI for downloading models and starting a local OpenAI-compatible server.

1. Install LM Studio.
2. Download a vision-capable instruct model.
3. Open the Developer or Local Server tab.
4. Load the model and start the server on port `1234`.
5. Keep this in `config.json`:

```json
{
  "local_llm_url": "http://127.0.0.1:1234/v1/chat/completions"
}
```

LM Studio's default OpenAI-compatible base URL is `http://localhost:1234/v1`, so this app points directly at its chat completions route.

### Command Line / Headless: llama.cpp

Use llama.cpp when you need a command-line server, especially on a Linux headless VPS or a dedicated inference box. Build or install llama.cpp, download a vision-capable GGUF model, then run `llama-server` on port `1234`:

```bash
llama-server -m /path/to/model.gguf --host 127.0.0.1 --port 1234 --ctx-size 4096
```

For remote Linux servers, prefer an SSH tunnel instead of exposing the LLM server directly to the public internet:

```bash
ssh -L 1234:127.0.0.1:1234 user@your-server
```

Then leave `config.json` pointed at `http://127.0.0.1:1234/v1/chat/completions` on the machine running Robot Scouter.

If your vision model requires a separate multimodal projector or special llama.cpp flags, follow the model's own llama.cpp instructions. The important part is that the final endpoint must accept OpenAI-compatible `POST /v1/chat/completions` requests with image input.

## Troubleshooting

### `SAM 3 initialization failed`

- Confirm `sam3.1_multiplex.pt` exists in the project root.
- Confirm `torch.cuda.is_available()` returns `True`.
- Confirm your `ultralytics` install supports `SAM3SemanticPredictor`.
- Select `HSV` in the Fuel Detector control if you need to keep working without SAM.

### `LMStudio not available`

- Start LM Studio's local server or `llama-server`.
- Confirm the URL in `config.json` ends with `/v1/chat/completions`.
- Use a vision-capable model.
- If the LLM runs on another machine, use an SSH tunnel or set `local_llm_url` to that host's reachable URL.

### OCR is disabled or inaccurate

- Install Tesseract OCR.
- On Linux, make sure `tesseract` is on `PATH`.
- On Windows, install to the default Tesseract path or add it to `PATH`.
- Use a clear 720p or 1080p source video.

### YouTube download fails

- Update dependencies:

```bash
pip install --upgrade yt-dlp static-ffmpeg
```

- Make sure the video is public or otherwise accessible from the machine running the app.

### Video output fails

- Install system FFmpeg and ensure `ffmpeg` is on `PATH`.
- Try a shorter clip with `Start Time` and `End Time` to confirm the workflow.
