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

- Install uv system-wide

```bash
make -s install-uv
```

- Install the project dependencies

```bash
make -s install
```

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
