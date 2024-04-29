# AI Benchmark

This works with latest CUDA 12.4 at the moment of writing this. Tested on Ubuntu 22.04

## Get Started

Make sure to have CUDA libraries in Path

```bash
export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}						 
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64\ {LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.9.0-0-Linux-x86_64.sh
chmod 777 Miniconda3-py38_23.9.0-0-Linux-x86_64.sh
bash Miniconda3-py38_23.9.0-0-Linux-x86_64.sh -b  -p ~/miniconda/
source ~/miniconda/bin/activate
conda init
conda config --set auto_activate_base false
conda deactivate
conda update -y conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

## Get Started Generic GPU with DirectML Support

To create environment follow this steps:

1. Create virtual env.

```bash
conda env create -v -f env-wsl2-directml.yml
```

2. Activate env
```bash
conda activate tensorflow-gpu
```

3. Run AI Benchmark

```bash
python run_ai_bench.py
```

## Get Started NVIDIA GPU

To create environment follow this steps:

1. Install miniconda if not installed:

2. Create virtual env.

```bash
conda env create -v -f env-wsl2-nvidia.yml
```

3. Activate env
```bash
conda activate tensorflow-gpu-nvidia
```

4. Run AI Benchmark

```bash
python run_ai_bench.py
```

## Notes

To remove env do:

```bash
conda deactivate
conda env remove -y --name tensorflow-gpu
```

or

```bash
conda deactivate
conda env remove -y --name tensorflow-gpu-nvidia
```