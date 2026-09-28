# Evaluation Task Notes

This project uses a predefined list of evaluation tasks for LLM benchmarking.

Note that the evaluation module of xLLM is preliminarily designed for ***debugging*** model training on commonly used downstream benchmarks.
To evaluate model checkpoints on advanced tasks, such as agentic benchmarks, 
please convert xLLM checkpoints to HF via [xBridges](https://github.com/ifm-ai/xbridges) and launch evaluation with advanced inference engine, 
such as [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang).
