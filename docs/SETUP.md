## Setup

Create a virtual environment named `venv`, activate it, and install dependencies.

**macOS / Linux**

```bash
cd cpsc440-project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (Command Prompt)**

```cmd
cd cpsc440-project
py -3 -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
cd cpsc440-project
py -3 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Notes:

- If you need a GPU build of PyTorch, install it from [pytorch.org](https://pytorch.org) for your platform, then install the rest with `pip install -r requirements.txt`.
- **SpeechBrain HiFi-GAN** downloads weights from Hugging Face; keep `speechbrain>=1.1.0`, `huggingface_hub>=0.25.0`, and matching **`torchaudio`** (see `requirements.txt`). On some setups, `torchaudio.list_audio_backends` is missing; `vocode.py` patches that before importing SpeechBrain.
- `matplotlib` may open plot windows when a script finishes; close them to return to the shell.

