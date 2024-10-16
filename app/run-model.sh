#!/bin/bash -x
pip install --upgrade pip
if [ "$(uname -i)" = "x86_64" ]; then
  if [ "$DEVICE" == "xla" ]; then
    pip install matplotlib Pillow -U
    pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
    pip install "optimum[neuronx, diffusers]"
  elif [[ "$DEVICE" == "cuda" || "$DEVICE" == "triton" ]]; then
    pip install environment_kernels
    pip install diffusers transformers accelerate safetensors matplotlib Pillow ipython torch -U
    pip install click nvitop
    pip install torch torchvision --upgrade --extra-index-url https://download.pytorch.org/whl/cu121
    python -c "import torch; print(torch.__version__)"
    python -c "import torchvision; print(torchvision.__version__)"
  fi
  uvicorn run-sd2:app --host=0.0.0.0
fi
