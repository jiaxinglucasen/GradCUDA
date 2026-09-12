Created project folder micrograd-cuda, opened in VS Code
Installed VS Code extensions: Python, C/C++
Installed Python 3.13, checked "Add to PATH"
Installed "Build Tools for Visual Studio" (Desktop development with C++ workload) — needed to compile C++/CUDA on Windows
Installed NVIDIA CUDA Toolkit — verified with nvcc --version
Created virtual environment: python -m venv venv, activated with venv\Scripts\activate
Installed NumPy — worked fine
First PyTorch install attempt failed: cu121 index has no wheels for Python 3.13
Fixed by switching to cu124 index:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
Created .gitignore (ignoring venv/, __pycache__/, *.pyc)
Sanity check passed: torch.cuda.is_available() → True
Used Ctrl+Shift+P → "Python: Select Interpreter" → chose the venv interpreter