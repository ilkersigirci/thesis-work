# Install

## Env File

```env
PUID=<REDACTED>
PGID=<REDACTED>
DAGSTER_HOME=<REDACTED>
WANDB_USER_NAME=<REDACTEd>
WANDB_API_KEY=<REDACTED>
```

## Default installation

- Install poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

- Install the project dependencies

```bash
conda create -n thesis-work python=3.10 -y
conda activate thesis-work
make -s install
```

- After running above command, the project installed in editable mode with all development and test dependencies installed.
- Moreover, a dummy `entry point` called `placeholder` will be available as a cli command.

### GPU Installations

#### CUDA

- `nvcc -V`

```bash
function lib_installed() { /sbin/ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep $1; }
function check() { lib_installed $1 && echo "$1 is installed" || echo "ERROR: $1 is NOT installed"; }

# Check if cuda is installed
check libcuda
check libcudart

# Check if cudnn is installed
check libcudnn
```

#### cuDNN

```bash
# Download cuDNN - 8.8.0.121 for CUDA 11.8
wget -O "cudnn-local-repo-ubuntu2204-8.8.0.121_1.0-1_amd64.deb" "https://developer.download.nvidia.com/compute/redist/cudnn/v8.8.0/local_installers/11.8/cudnn-local-repo-ubuntu2204-8.8.0.121_1.0-1_amd64.deb"

# Install cuDNN
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.8.0.121_1.0-1_amd64.deb

# Install keyring
sudo cp /var/cudnn-local-repo-ubuntu2204-8.8.0.121/cudnn-local-B66125A0-keyring.gpg /usr/share/keyrings/

# Install related deb packages
sudo apt-get update
sudo apt-get install libcudnn8=8.8.0.121-1+cuda11.8
sudo apt-get install libcudnn8-dev=8.8.0.121-1+cuda11.8
sudo apt-get install libcudnn8-samples=8.8.0.121-1+cuda11.8
```

#### Tensorflow GPU Setup

- For latest ubuntu CUDA installation, `sudo ln -s /usr/lib/cuda /usr/local/cuda` might necessary.
    - But this probably **breaks** pytorch installation. To reverse it: `sudo unlink /usr/local/cuda`
- As stated in [tensorflow documentation](https://www.tensorflow.org/install/pip), `CUDNN_PATH` should be in `LD_LIBRARY_PATH` in order to use GPU with tensorflow.
- To do so with conda,

```bash
conda activate thesis-work
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

- libcuda error fix

```bash
# WORKING but is there any other way?
sudo ln -s $CONDA_PREFIX/nvvm/libdevice/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/libdevice.10.bc

# Alternative - NOT WORKING
echo 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

## Docker

```bash
docker build --tag thesis-work --file docker/Dockerfile --target development .

docker build --tag thesis-work --file docker/Dockerfile --target production .
```

- To run command inside the container:

```bash
docker run --rm -it thesis-work:latest bash

# Temporary container
docker run -it thesis-work:latest bash
```

# Useful Makefile commands

```bash
# All available commands
makefile
makefile help

# Run all tests
make -s test

# Run specific tests
make -s test-one TEST_MARKER=<TEST_MARKER>

# Remove unnecessary files such as build,test, cache
make -s clean

# Run all pre-commit hooks
make -s pre-commit

# Lint the project
make -s lint

# Profile a file
make -s profile PROFILE_FILE_PATH=<PATH_TO_FILE>
```
