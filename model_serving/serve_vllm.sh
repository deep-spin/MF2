#!/usr/bin/env bash
set -e

############################
# User-defined variables
############################

# dionysus and poseidon venv paths
# VENV_PATH="/mnt/scratch-dionysus/manos/venvs/venv-f3-glm" #use this environment for zai-org/GLM-4.5V-FP8, zai-org/GLM-4.5V
# VENV_PATH="/mnt/data-poseidon/manos/venvs/venv-f3-ernie" #use this environment for baidu/ERNIE-4.5-VL-28B-A3B-Thinking

#hades venv paths
# VENV_PATH="/mnt/scratch-hades/manos/venvs/venv-f3-vllm"
# VENV_PATH="/mnt/scratch-hades/manos/venvs/venv-f3-glm" #use this environment for hosting: zai-org/GLM-4.5V-FP8, zai-org/GLM-4.5V 
VENV_PATH="/mnt/scratch-hades/manos/venvs/venv-f3-ernie" #use this environment for hosting: baidu/ERNIE-4.5-VL-28B-A3B-Thinking


############################
# Activate environment
############################

source "${VENV_PATH}/bin/activate"

############################
# Launch vLLM server
############################

## GLM-4.5V-FP8 and GLM-4.5V models: Run this model on dionysus with 2 GPUs. If you want to run zai-org/GLM-4.5V you need to run it on hades with 2 GPUs.
# Remember to set the VENV_PATH variable to the correct environment.

# exec vllm serve zai-org/GLM-4.5V-FP8 \
#     --host "hades" \
#     --port "8000" \
#     --tensor-parallel-size 2 \
#     --tool-call-parser glm45 \
#     --reasoning-parser glm45 \
#     --enable-auto-tool-choice \
#     --allowed-local-media-path / \
#     --media-io-kwargs '{"video": {"num_frames": 180, "fps": 1}}' \
#     --max-num-seqs 1 \
#     --served-model-name glm-4.5v-fp8


# ERNIE-4.5-VL-28B-A3B-Thinking model: Run this model on dionysus with 2 GPUs.
# Remember to set the VENV_PATH variable to the correct environment.

exec vllm serve baidu/ERNIE-4.5-VL-28B-A3B-Thinking \
    --host "hades" \
    --port "8003" \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --reasoning-parser ernie45  \
    --tool-call-parser ernie45  \
    --enable-auto-tool-choice \
    --allowed-local-media-path / \
    --served-model-name ernie-4.5-vl \
    --max-num-seqs 1 \
    --media-io-kwargs '{"video": {"num_frames": 120, "fps": 1}}'