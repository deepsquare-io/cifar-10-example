# CIFAR 10 Horovod Example

This example uses the [Deep Layer Aggregation](https://arxiv.org/abs/1707.06484) method to train on the CIFAR10 dataset.

## Installation with Pipenv

1. Install OpenMPI if you wish to be able to run a distributed workload locally.

2. Install [Pipenv](https://pipenv.pypa.io/en/latest/) which is a dependency management tool with a locking mechanism (similar to Anaconda).

3. Clone this repository and run:

   ```shell
   export HOROVOD_WITH_PYTORCH=1
   export HOROVOD_WITH_MPI=1
   export HOROVOD_WITHOUT_GLOO=1

   # If GPU
   # export HOROVOD_CUDA_HOME=/usr/local/cuda
   # export HOROVOD_GPU=CUDA
   pipenv install
   ```

   This command creates a virtualenv based on the Pipfile and Pipfile.lock.

## Usage

### With Docker

Prepare the directories:

```shell
mkdir -p "$(pwd)/data"
# Download CIFAR-10 dataset
curl -fsSL https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -o "$(pwd)/data/cifar-10-python.tar.gz"
tar -C $(pwd)/data/ -xvzf "$(pwd)/data/cifar-10-python.tar.gz"
mkdir -p "$(pwd)/checkpoint"

```

Run the model:

```shell
docker run \
  --rm \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/checkpoint:/checkpoint" \
  -u 1000:1000 \
  --entrypoint /bin/sh \
  ghcr.io/deepsquare-io/cifar-10-example:latest \
  -c '\
  mpirun \
  -np 4 \
  /.venv/bin/python3 \
  /app/main.py \
  --no-cuda \
  --horovod \
  --checkpoint_in=/checkpoint/ckpt.pth \
  --checkpoint_out=/checkpoint/ckpt.pth \
  --dataset=/data
'
```

### With Pipenv

Prepare the directories:

```shell
mkdir -p "$(pwd)/data"
# Download CIFAR-10 dataset
curl -fsSL https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -o "$(pwd)/data/cifar-10-python.tar.gz"
tar -C $(pwd)/data/ -xvzf "$(pwd)/data/cifar-10-python.tar.gz"
mkdir -p "$(pwd)/checkpoint"

```

Run the model:

```shell
pipenv shell
mpirun \
  -np 4 \
  python3 \
  main.py \
  --no-cuda \
  --horovod \
  --checkpoint_in="$(pwd)/checkpoint/ckpt.pth" \
  --checkpoint_out="$(pwd)/checkpoint/ckpt.pth" \
  --dataset="$(pwd)/data"
'
```
