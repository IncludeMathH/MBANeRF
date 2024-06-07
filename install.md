# Intall [nerfstudio](https://docs.nerf.studio/quickstart/installation.html)

## create environment
```powershell
conda create --name nerfstudio -y python=3.10
conda activate nerfstudio
python -m pip install --upgrade pip
```

## Dependencies
### Pytorch
```powershell
# from conda
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# from pip
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# build teh necessary CUDA extensions
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

### tiny-cuda-nn
```powershell
sudo apt-get install build-essential git
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn

cd tiny-cuda-nn
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j

cd bindings/torch
python setup.py install
```

## Installing nerfstuio
```powershell
# from pip
pip install nerfstudio

# from source
git clone https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
```
# Install vim
```powershell
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal_conv1d
git checkout v1.1.0
# edit setup.py to add the lines here when using Titan-X or P100 GPU:
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_60,code=sm_60")
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .

cd Vim
# modifi setup.py in line 274:
"causal_conv1d>=1.1.0" -> "causal_conv1d==1.1.0"
# edit setup.py to add the lines here when using Titan-X or P100 GPU:
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_60,code=sm_60")
pip install -e mamba-1p1p1
```

# Install MBANeRF
```powershell
cd MBANeRF/
pip install -e .
ns-install-cli
```