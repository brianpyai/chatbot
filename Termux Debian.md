Here's the modified markdown with copy buttons added to each code block:

####Debian:

```bash
termux-setup-storage
termux-wake-lock
termux-change-repo
pkg install tur-repo
pkg update && pkg upgrade 
pkg install git proot-distro termimage vim termux-api
```

```bash
disname='debian'
user='brian'
echo "proot-distro login $disname"  > .shortcuts/debianai.sh
echo  "proot-distro login debian --user $user" > .shortcuts/debian.sh
proot-distro install debian
proot-distro login debian
```

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

```bash
echo 'source llm/bin/activate && source .cargo/env &&  cd /sdcard/Documents/Pydroid3/llm' > llm.env
sudo locale-gen zh_CN.UTF-8
sudo apt update && apt upgrade
sudo apt install python3-full git curl vim wget
sudo apt install python-is-python3
sudo apt install python3-pip
sudo apt install  clang wget git cmake
sudo apt install  opencl-headers
sudo apt install  libopenblas-dev libopenblas0 libopenblas64-0 libblis-openmp-dev
sudo apt install python3-torch python3-torchaudio python3-torchtext python3-torchvision
sudo apt install libideep-dev libtorch-dev libonnx1
sudo apt install pandoc build-essential
sudo apt install libopenblas-dev libopenblas-openmp-dev libopenblas0 libopenblas64-0-openmp libopenblas64-dev libopenblas64-openmp-dev 
sudo apt install opencl-c-headers opencl-clhpp-headers libasl-dev libasl0 libclblast-dev libclc-13 
sudo apt install pandoc git-lfs
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
git clone --recursive https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp
mkdir build
cd build
cmake .. -DGGML_OPENBLAS=ON
cmake --build . --config Release
```

```bash
rm -rf llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build 
make GGML_OPENBLAS=1
cd
python -m venv llm
source llm/bin/activate
```

```bash
pip install --upgrade pip
pip uninstall ctransformers llama-cpp-python
rm -rf ctransformers
git clone --recurse-submodules https://github.com/marella/ctransformers
cd ctransformers 
bash ./scripts/build.sh
cd 
python -m pip install ./ctransformers/
CMAKE_ARGS="-DLLAMA_BLAS=ON" pip install llama-cpp-python 
```

```bash
pip install --upgrade diffusers[torch] 
pip install --upgrade  accelerate peft openvino optimum onnx onnxruntime nncf
pip install opencv-python fastapi uvicorn flask
pip install fastapi python-multipart pydantic sqlalchemy opencc-python-reimplemented pandas 
pip install --upgrade pip
pip install fast-sentence-transformers
pip install langchain
pip install wikipedia  unstructured pypdf pdf2image pdfminer chromadb qdrant-client lark momento annoy
pip install doc2text pypandoc pandoc
pip install opencv-python fastapi uvicorn flask
pip install fastapi python-multipart pydantic sqlalchemy opencc-python-reimplemented pandas gradio 
pip install --upgrade-strategy eager install optimum[onnxruntime]
pip install optimum-intel
pip install --upgrade-strategy eager optimum[openvino,nncf]
pip install --force-reinstall transformers[torch]
pip install --force-reinstall diffusers[torch] 
```

