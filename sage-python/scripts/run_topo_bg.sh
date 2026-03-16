#!/bin/bash
# Background TopologyBench runner -- fully detached
export HF_HUB_OFFLINE=1
export PYTHONIOENCODING=utf-8
export REQUESTS_CA_BUNDLE=""
cd "C:/Code/YGN-SAGE/sage-python"

# If resume file exists, use it
if [ -f "data/topologybench_164_real.json" ]; then
    echo "Resuming from partial results..."
    python scripts/run_topologybench.py \
        --tasks 164 \
        --topologies sequential,debate,brainstorming,parallel \
        --resume data/topologybench_164_real.json \
        --output data/topologybench_164_real.json \
        >> data/topologybench_164_real.log 2>&1
else
    echo "Starting fresh run..."
    python scripts/run_topologybench.py \
        --tasks 164 \
        --topologies sequential,debate,brainstorming,parallel \
        --output data/topologybench_164_real.json \
        >> data/topologybench_164_real.log 2>&1
fi

echo "EXIT CODE: $?" >> data/topologybench_164_real.log
echo "DONE at $(date)" >> data/topologybench_164_real.log
