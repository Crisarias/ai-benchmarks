# AI Benchmarks

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

# AI Bench Alpha

## Get Started Generic GPU with DirectML Support

To create environment follow this steps:

1. Create virtual env.

```bash
conda env create -v -f env-wsl2-tf-directml.yml
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

1. Create virtual env.

```bash
conda env create -v -f env-wsl2-tf-nvidia.yml
```

2. Activate env
```bash
conda activate tensorflow-gpu-nvidia
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
# Verify GPUs available
python3 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))"

```

3. Run AI Benchmark

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

# RESNET 50 with CIAT10 Data set

## Get Started NVIDIA GPU

To create environment follow this steps:

1. Create virtual env.

```bash
conda env create -v -f env-wsl2-pytorch-nvidia.yml
```

3. Activate env
```bash
conda activate pytorch-gpu-nvidia
# Verify GPUs available
python3 -c "import torch; print(torch.cuda.is_available())"
```

4. Run Training on one node

Based on https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide. Find best batch size and epoch

```bash
cd LambdaLabsML-examples/pytorch/distributed/resnet
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=10.1.96.5 --master_port=1234 main.py --backend=nccl --batch_size=1024 --num_epochs=50 --arch=resnet50
```

5. Run training distributed

```bash
mpirun -env MASTER_ADDR=10.1.96.5 -env MASTER_PORT=1234 -envall -bind-to none -map-by slot python3 main.py --backend=nccl --use_syn --batch_size=1024 --num_epochs=50 --arch=resnet50
```
## Notes

Profiling python -m cProfile myscript.py

