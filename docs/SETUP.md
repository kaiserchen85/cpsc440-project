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
- `matplotlib` may open plot windows when a script finishes; close them to return to the shell.

