
1. Install Termux, Termux:Widget, and Termux:API from F-Droid.

2. Open Termux and run initial setup commands:
```bash
termux-setup-storage
termux-wake-lock
termux-change-repo
pkg install tur-repo
pkg update && pkg upgrade 
pkg install git proot-distro vim termux-api
```

3. Install and set up Debian using proot-distro:
```bash
disname='debian'
user='brian'
echo "proot-distro login $disname"  > .shortcuts/debianai.sh
echo  "proot-distro login debian --user $user" > .shortcuts/debian.sh
proot-distro install debian
proot-distro login debian
```

4. After logging into Debian, set up the environment:
```bash
user='brian'
apt update && apt upgrade
apt install sudo locales
echo "LANG=zh_CN.UTF-8" >> /etc/locale.conf
sudo locale-gen
adduser $user
gpasswd -a $user sudo
echo "$user   ALL=(ALL:ALL) ALL" >> /etc/sudoers
login $user
```

5. Install necessary packages and set up Python environment:
```bash
sudo apt update && apt upgrade
sudo apt install python3-full git curl vim wget python-is-python3 python3-pip
sudo apt install clang cmake opencl-headers libopenblas-dev
sudo apt install python3-torch python3-torchaudio python3-torchtext python3-torchvision
sudo apt install libideep-dev libtorch-dev libonnx1
sudo apt install pandoc build-essential git-lfs
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
git clone https://github.com/RustPython/RustPython
cd RustPython
cargo install --path . --features ssl
rustpython --install-pip
cd
echo "source rustpy/bin/activate && source .cargo/env &&  cd /sdcard/Documents/Pydroid3/llm" > rustpy.env
rustpython -m venv rustpy
source rustpy.env

```

6. Build Llama with OPENBLAS:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build 
make GGML_OPENBLAS=1
cd
python -m venv llm
echo "source llm/bin/activate && source .cargo/env &&  cd /sdcard/Documents/Pydroid3/llm" > llm.env
source llm/bin/activate
```

7. Install llama-cpp-python and other Python packages:
```bash
pip install --upgrade pip
CMAKE_ARGS="-DLLAMA_BLAS=ON" pip install llama-cpp-python 
pip install --upgrade diffusers[torch] accelerate peft openvino optimum onnx onnxruntime nncf
pip install opencv-python fastapi uvicorn flask
pip install fastapi python-multipart pydantic sqlalchemy opencc-python-reimplemented pandas 
pip install fast-sentence-transformers langchain
pip install wikipedia unstructured pypdf pdf2image pdfminer chromadb qdrant-client lark momento annoy
pip install doc2text pypandoc pandoc gradio
```

This setup provides a Debian environment within Termux, with Python and various AI-related libraries installed. The environment is ready for development and running AI models, including Llama.

