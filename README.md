# Install

## Env File

```env
PUID=<REDACTED>
PGID=<REDACTED>
DAGSTER_HOME=<REDACTED>
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

## Docker

```bash
# Development build (800 MB)
docker build --tag thesis-work --file docker/Dockerfile --target development .

# Production build (145 MB)
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
