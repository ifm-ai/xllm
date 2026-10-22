# Team instructions (from the user)

## The project
xLLM: an LLM training framework in this repository. Read the code to learn how it trains, launches and checkpoints.

## Requirements
- NCCL-only process groups: no gloo or other CPU backend.
- Generic, mostly for NVIDIA GPU clusters (CUDA, NCCL): not tied to one particular cluster, scheduler or filesystem.
- Any model size and any cluster size: no upper bound on model size.

## How to work
- Read the code before claiming anything; cite file:line for every claim. Citations are checked automatically.
- You cannot run code. The user will run tests later.
- If something essential is unclear, ask with ASK_USER: (the user may answer during the run); otherwise state your assumption.
- Do not finish until the other member agrees.
