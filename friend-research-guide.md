# Your research workspace

## Connect from your Windows laptop

Keep Tailscale connected, then run in PowerShell:

```powershell
ssh -F "$env:USERPROFILE\.ssh\research-pc.conf" research-pc
```

This opens Ubuntu on the research PC. Your files belong in `/workspace`.

```bash
cd /workspace
source /workspace/.venv/bin/activate
python --version
```

Resources: **8 virtual CPUs, up to 24 GiB RAM, and a 500 GiB virtual disk**. Linux and tools occupy part of the disk. This environment is **CPU only: CUDA/GPU access is not available**.

Python, CPU PyTorch, JupyterLab, NumPy, SciPy, pandas, matplotlib, scikit-learn, Git, Git LFS, a compiler, and tmux are installed. Add Python packages with `python -m pip install PACKAGE`. Use HTTPS URLs for repositories and datasets; public web downloads go through the configured proxy.

## Keep a job running after disconnecting

```bash
tmux new -s research
# Run your program inside tmux.
```

Press Ctrl+B, then D, to detach. After reconnecting, use `tmux attach -t research`.

## Upload and download

Run these on your Windows laptop:

```powershell
scp -F "$env:USERPROFILE\.ssh\research-pc.conf" .\train.py research-pc:/workspace/
scp -F "$env:USERPROFILE\.ssh\research-pc.conf" research-pc:/workspace/result.csv .
```

## Jupyter

Connect with the notebook port forwarded:

```powershell
ssh -F "$env:USERPROFILE\.ssh\research-pc.conf" -L 127.0.0.1:8888:127.0.0.1:8888 research-pc
```

Inside Ubuntu, run `research-jupyter`, optionally inside tmux. Open the localhost URL with the token it prints. Keep your SSH connection open while using the browser.

You have no sudo access or access to the owner's Windows files, LAN, or Tailscale devices. Ask the owner for system packages that require administrator installation. Keep your research private key on your laptop.

Both the owner's Fedora laptop and Windows PC need to stay on and awake. The owner can pause access or stop the VM when the PC's resources are needed.
