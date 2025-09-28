# CNN project — setup and dataset download

This repository contains code and data for training a CNN on chest X-ray images. This README explains how to create a Python virtual environment on Windows (PowerShell), install required packages from `requirements.txt`, and download the dataset from Google Drive using `gdown` (CLI and Python examples).

## Prerequisites

- Windows with PowerShell (instructions use PowerShell commands)
- Python 3.8 or newer installed and available as `python` on PATH
- pip
- At least several gigabytes of disk space for the dataset

If you don't have Python installed, download it from https://www.python.org/downloads/ and enable "Add Python to PATH" during installation.

## 1) Create and activate a virtual environment (PowerShell)

Open PowerShell in the repository folder (for example `D:\Programs\CNN`) and run:

```powershell
# create a venv in a folder named .venv
python -m venv .venv

# on PowerShell, activate the venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks execution of the activation script because of the execution policy, allow it for the current process only (no admin required):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1
```

Alternative (cmd.exe):

```powershell
# from PowerShell you can still run cmd-style activate
.\.venv\Scripts\activate
```

Once activated, your prompt should show the virtual environment name (for example `(.venv) PS D:\Programs\CNN>`).

## 2) Upgrade pip and install dependencies

With the virtual environment active, upgrade pip and install the requirements:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` does not include `gdown`, install it separately (next step shows this too).

## 3) Install gdown (for downloading from Google Drive)

gdown makes it easy to download files shared via Google Drive from scripts or the CLI.

```powershell
pip install gdown
```

## 4) Download the dataset from Google Drive using gdown

You will need the Google Drive file ID or a shareable link for the dataset (e.g. `chest_xray.zip`).

How to extract the file ID from a share link:

- A typical Drive share link looks like: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing` — copy the portion labeled `FILE_ID`.

CLI example (using file ID):
The File is 

```css
1ExZ-N4_RcMzNZO2uI-wo7pJqAFY135X9
```


```powershell
# replace <FILE_ID> with the actual id
gdown --id 1ExZ-N4_RcMzNZO2uI-wo7pJqAFY135X9 -O chest_xray.zip
```

Or using the full share URL:

```powershell
gdown "https://drive.google.com/uc?id=<FILE_ID>" -O chest_xray.zip
```

Python snippet (programmatic download):

```python
import gdown

url = 'https://drive.google.com/uc?id=<FILE_ID>'  # replace <FILE_ID>
output = 'chest_xray.zip'
gdown.download(url, output, quiet=False)
```

Notes:
- For very large files Drive may present a confirmation page for virus scanning; `gdown` handles this automatically in most cases.
- If the file is in a shared folder or requires special access, ensure the file is shared publicly or that you are signed in with the correct account.

## 5) Extract the dataset

After download, extract the zip into the repository root (PowerShell):

```powershell
Expand-Archive -Path .\chest_xray.zip -DestinationPath . -Force
```

This should create a `chest_xray/` folder with `train/`, `val/`, and `test/` subfolders.

If you prefer to use a Python extraction snippet:

```python
import zipfile

with zipfile.ZipFile('chest_xray.zip', 'r') as z:
	z.extractall('.')
```

## 6) Verify the dataset

Check the folder structure after extraction. Example expected layout:

- `chest_xray/train/NORMAL/` and `chest_xray/train/PNEUMONIA/`
- `chest_xray/val/`
- `chest_xray/test/`

You can list files in PowerShell to spot-check:

```powershell
Get-ChildItem -Directory -Recurse chest_xray | Select-Object -First 20
```

## Troubleshooting

- ExecutionPolicy errors when activating venv: use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force` as shown above.
- `gdown` not found: ensure the venv is activated before installing and running `gdown`.
- Permission errors installing packages: run PowerShell as admin only if required, but prefer virtual environments to avoid system installs.
- Out of disk space during download or extraction: ensure you have enough free space for the zip and extracted contents.

## Optional: make a small helper script

You can add a small helper script (for example `scripts/download_dataset.py`) that calls `gdown` and extracts the zip so others can reproduce the steps with a single command.

## Files changed

- `README.md` — added installation, dataset download, and extraction instructions (PowerShell-friendly).

---

If you'd like, I can:

- add a small `download_dataset.py` helper that reads a file ID from an environment variable or CLI and runs `gdown` + extraction, or
- update `requirements.txt` to include `gdown` automatically.

Tell me which of those you'd like and I'll implement it.
