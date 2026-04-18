#!/bin/bash
set -e

export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

pip3 install openai -i --default-timeout 0.3 https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

cd /workspace
python3 main.py /target/target_spec.json /workspace/output.json
