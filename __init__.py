"""
qwen_fast - public API.

Drop-in fast inference for Qwen 2.5 on consumer GPUs (Windows-native).
Output is byte-identical to vanilla HF generate (cos = 1.0).

Public surface:
    FastQwen(model_id, device='cuda')  -- engine class
    FastQwen.generate(input_ids, max_new_tokens, ...) -> output_ids
    FastQwen.benchmark(prompts, gen_len) -> dict of timings

The engine implementation is proprietary; this module exposes only the
user-facing API.
"""

from ._internal.engine import FastQwen

__version__ = "0.1.0"
__all__ = ["FastQwen", "__version__"]
