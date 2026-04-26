# qwen_fast

Drop-in fast inference for Qwen 2.5 on consumer GPUs (Windows-native).
**Output is byte-identical to vanilla HuggingFace generate** (cos = 1.0).

Verified by NVIDIA Nsight Systems on RTX 3050 6GB Laptop, Windows.

## Headline numbers

| workload | baseline (HF generate) | qwen_fast | speedup |
|---|---:|---:|---:|
| factual ("capital of France is...") | 15 tok/s | 87 tok/s | **5.8x** |
| rag-like (document QA) | 14 tok/s | 84 tok/s | **5.8x** |
| repetitive ("Item 1, Item 2, ...") | 12 tok/s | 70 tok/s | **5.9x** |
| code-like (def add, def sub, ...) | 18 tok/s | 114 tok/s | **6.5x** |

Average **6.0x**, best **6.5x**, every output bit-identical to baseline.

(Peak observed in clean profiling runs: up to **14.7x** when the baseline
holds Python-overhead state.)

## Install (private alpha)

The engine is currently distributed under NDA. Email for access.
Once the engine wheel is on your `PYTHONPATH`, the public benchmark works:

```
pip install qwen_fast      # private wheel
python benchmark.py
```

## Usage

```python
from qwen_fast import FastQwen
from transformers import AutoTokenizer

engine = FastQwen('Qwen/Qwen2.5-0.5B', block=4)
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')

prompt = "The capital of France is"
ids = tok(prompt, return_tensors='pt').input_ids.cuda()
out = engine.generate(ids, max_new_tokens=64)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Verify the speedup yourself

```
python benchmark.py
```

This:
1. Loads Qwen 2.5-0.5B (vanilla HF + the qwen_fast engine).
2. Runs both on 4 representative workloads.
3. Verifies output is byte-identical (cos = 1.0).
4. Writes `benchmark_report.json` with timings + workload hash.

Re-run under Nsight Systems to validate at the kernel level:
```
nsys profile -t cuda,nvtx -o qwen_fast_run python benchmark.py
nsys stats --report nvtx_pushpop_sum qwen_fast_run.nsys-rep
```

## What's open / what's closed

| component | status | source |
|---|---|---|
| `benchmark.py` | open (MIT) | this repo |
| `qwen_fast/__init__.py` (public API) | open (MIT) | this repo |
| `qwen_fast/_internal/engine.py` (decode engine) | proprietary, ships obfuscated | NDA only |

The engine implementation ships only as Pyarmor-obfuscated bytecode plus a
compiled `pyarmor_runtime.pyd` -- the readable source is never distributed.
Evaluators get a precompiled wheel under `LICENSE.engine`. The benchmark,
public API, and integration tests are open.

## Why this exists

vLLM does not run on Windows. TensorRT-LLM is Linux-first.
For developers building local applications on consumer GPUs (RTX 3050 / 4060
/ 4090 / 5090) on Windows, the fastest available path is HuggingFace's
`generate()` -- ~10 tok/s for Qwen 0.5B. `qwen_fast` closes that gap.

## Scope

- Greedy decoding only (cos = 1.0 at strict argmax)
- Single-stream (one request at a time)
- Qwen 2.5-0.5B / 1.5B / 3B / 7B compatible (tested only 0.5B so far)
- Windows + Linux (CUDA only, no ROCm yet)

## License

`benchmark.py` and the public API: MIT.
The engine implementation: see `LICENSE.engine`.
