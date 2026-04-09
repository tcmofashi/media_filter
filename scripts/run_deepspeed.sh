#!/bin/bash
# DeepSpeed environment setup
if [ -z "${CUDA_HOME:-}" ]; then
  if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/bin" ]; then
    export CUDA_HOME="${CONDA_PREFIX}"
  else
    export CUDA_HOME="/usr/local/cuda"
  fi
fi

export PATH="${CUDA_HOME}/bin:${PATH}"

if [ -d "${CUDA_HOME}/lib64" ]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
elif [ -d "${CUDA_HOME}/lib" ]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
fi

# Run the command
"$@"
