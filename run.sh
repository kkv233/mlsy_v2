#!/bin/bash
set -e

# Install dependencies if needed
pip3 install openai -i --default-timeout 0.3 https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Run the agent
# Input: /target/target_spec.json
# Output: /workspace/results.json
cd /workspace
python3 main.py /target/target_spec.json /workspace/results.json
