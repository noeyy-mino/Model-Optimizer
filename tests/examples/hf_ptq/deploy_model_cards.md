# Model-card deployment cases

Reviewed on 2026-11-12. The seven checkpoints below add 14 explicit model/backend
cases to `test_deploy.py::test_model_card_deploy`. Configuration lives in
`deploy_model_cards.json`; the runner is `tests/_test_utils/hf_serving.py`.
These are serving smoke tests, not accuracy, performance, or speculative acceptance-rate benchmarks.

## Cases and source evidence

| Checkpoint / source | Added backends | Recipe selected |
| --- | --- | --- |
| [DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) | vLLM, SGLang | TP4, Blackwell; MXFP4 Engram tables offloaded to host memory; backend-specific attention and KV-cache settings |
| [Muse-Spark-Open-816B-A42B-DFlash2](https://huggingface.co/nvidia/Muse-Spark-Open-816B-A42B-DFlash2) | vLLM, SGLang | TP4/EP4, Blackwell; paired target + DFlash2 draft, never standalone draft inference |
| [Nemotron-3.5-Super-VL BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16) | vLLM, SGLang | Four-GPU Hopper recipe; native MTP, no forced FP4 quantization |
| [Nemotron-3.5-Super-VL NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4) | vLLM, SGLang, TensorRT-LLM | Two-GPU Hopper recipe for vLLM/SGLang; one-GPU Blackwell recipe for TensorRT-LLM; native MTP |
| [Nemotron LiLiCorr](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr) | vLLM, SGLang, TensorRT-LLM | One GB10, paired NVFP4 target + LiLiCorr draft; DGX Spark recipes from the target card; explicitly skips other GPU names |
| [Step-5-Preview-NVFP4](https://huggingface.co/stepfun-ai/Step-5-Preview-NVFP4) | SGLang | TP4/EP4, Blackwell, FlashInfer FP4 MoE, BF16 KV cache, multi-layer MTP; no MegaMoE |
| [Qwen-Image-NVFP4](https://huggingface.co/nvidia/Qwen-Image-NVFP4) | TensorRT-LLM VisualGen | One Blackwell GPU, eager configuration, actual 512×512 PNG generation |

For Step-5, the HF README's deployment examples cover FP8/BF16. The
[SGLang Step-5 cookbook](https://docs.sglang.io/cookbook/autoregressive/StepFun/Step-5-Preview)
explicitly documents the NVFP4 checkpoint on Blackwell and rules out MegaMoE and
H200 for that precision. A vLLM NVFP4-specific recipe was not established from
these sources, so this addition does not assert vLLM support for this checkpoint.

The Muse vLLM recipe uses `meta-models/Muse-Spark-Open-816B-A42B`, exactly as its
card's vLLM example specifies; SGLang uses the separate
`meta-models/Muse-Spark-Open-816B-A42B-NVFP4` target. Do not confuse the small draft
checkpoint with the full target's GPU memory and download requirements.

## Required runtime builds

- DeepSeek vLLM: v0.31.0 with MXFP4 Engram PR #61530. SGLang: PR #47935,
  available from 2026-11-04. Host Engram tables require about 97 GiB per node.
- Muse SGLang: `lmsysorg/sglang:dev-muse-spark-open`. Its vLLM build must support
  Muse and DFlash2 with probabilistic draft sampling; the card does not pin an image.
- Nemotron vLLM: v0.32.0 or the nightly hash recorded in the JSON.
  SGLang: `lmsysorg/sglang:nemotron3-5-super`.
- Nemotron TensorRT-LLM: 1.3.0rc28 plus the Mamba cache patch from PR #19597.
- Step-5: `lmsysorg/sglang:step-5-preview`, including its FlashInfer patches.
- Qwen-Image: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc28`.

Run each backend's cases in its matching prepared environment. The runner checks
package/executable presence and GPU count/SM, not whether a vendor patch has been
applied. An installed but incompatible runtime fails with its startup log; it is
not silently reported as unsupported or passed. GPU SM/count checks do not prove
sufficient memory. DeepSeek/Muse examples were validated upstream on GB300,
Step-5 on B300, and VisualGen on GB300; other qualifying GPUs remain unvalidated.

## Scope and workload adaptations

Most text recipes reduce the maximum context to 4096 and concurrency to at most
four (one for Spark), with one short request and up to 128 output tokens. The
TensorRT-LLM Nemotron YAML retains its card's context/cache settings. These
reductions are smoke-test choices, not the model card's full-length benchmark.
Text smoke tests accept nonempty answer **or reasoning** with positive generated
token usage. They do not validate image/video understanding for the VLMs, tool
calling, answer correctness, or speculative speedup. The image case instead
requires one decodable PNG of exactly 512×512 pixels, using 50 steps and seed 42;
it does not evaluate image quality.

No automatic container installation or GPU allocation occurs. TRT-LLM cases
resolve complete local snapshots at execution time. Set `MODELOPT_LOCAL_MODEL_ROOT`
to use pre-staged checkpoints under `ROOT/organization/model`; include the target
and draft directories for paired cases. Otherwise the runtime/HF Hub uses the
normal authenticated cache. Collection never downloads weights. The new cases
preserve their model cache; the legacy cases retain their existing cleanup behavior.

## Run

From the repository root, inside the required runtime environment:

```bash
python -m pytest tests/examples/hf_ptq/test_deploy.py::test_model_card_deploy \
  --collect-only -q --run-release

python -m pytest tests/examples/hf_ptq/test_deploy.py::test_model_card_deploy \
  --run-release -v --capture=tee-sys -k 'Step-5-Preview-NVFP4 and sglang'
```

Do not run GPU cases concurrently on the same allocated devices. The runner uses
only the first required devices in `CUDA_VISIBLE_DEVICES` and binds to loopback.
Startup has a 3600-second budget, generation 600 seconds. The new tests default to
4800 seconds overall; an explicit `MODELOPT_QA_TEST_TIMEOUT` overrides that pytest
budget. Pre-stage large downloads so they do not exhaust the overall timeout.

The printed pytest temporary directory contains `command.json`, `server.log`,
`response.json`, optional `server-config.yaml`, and `generated.png` for VisualGen.
Archive that directory as a CI artifact. The runner tears down its process group
on success, startup failure, and request failure. A host kill or machine failure
still requires the CI/container supervisor to clean up.
