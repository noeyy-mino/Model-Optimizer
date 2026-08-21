#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LPIPS quality evaluation for quantized diffusion models.

Generates a BF16 golden image and a quantized image from the same prompt and
seed, then asserts that the LPIPS distance is below a configurable threshold.

Workflow:
  1. Load model in BF16 precision.
  2. Generate a golden image with a fixed seed (no quantization).
  3. Quantize the transformer backbone in-memory via mtq.
  4. Generate a quantized image with the same seed and prompt.
  5. Compute LPIPS between golden and quantized images.
  6. Print the score and exit 0 (PASS) or 1 (FAIL).

Usage:
    python lpips_eval.py --model flux-dev --format fp8
    python lpips_eval.py --model flux-dev --format fp4 --threshold 0.08
    python lpips_eval.py --model flux-schnell --format fp8 --override-model-path /models/FLUX.1-schnell
"""

import argparse
import copy
import logging
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF

from config import (
    FP8_DEFAULT_CONFIG,
    INT8_DEFAULT_CONFIG,
    NVFP4_DEFAULT_CONFIG,
    set_quant_config_attr,
)
from models_utils import MODEL_DEFAULTS, ModelType
from pipeline_manager import PipelineManager
from quantize_config import ModelConfig, QuantAlgo, QuantFormat

import modelopt.torch.quantization as mtq

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
LPIPS_PROMPT = "A serene mountain landscape with a clear blue sky and snow-capped peaks"
LPIPS_SEED = 42
LPIPS_HEIGHT = 256
LPIPS_WIDTH = 256
LPIPS_NUM_STEPS = 4
LPIPS_THRESHOLD = 0.05
LPIPS_CALIB_SIZE = 8
LPIPS_N_STEPS = 10

# Short fixed prompts used for calibration — avoids HuggingFace dataset download.
_CALIB_PROMPTS = [
    "A serene mountain landscape with a clear blue sky",
    "A busy city street at night with neon lights",
    "A colorful abstract painting with geometric shapes",
    "A portrait of a person with natural lighting",
    "A sunset over the ocean with warm golden tones",
    "A forest path covered with morning mist",
    "A modern kitchen interior with marble countertops",
    "A cat sitting on a windowsill watching the rain",
    "A red sports car parked on a coastal road",
    "A field of sunflowers under a bright blue sky",
    "An astronaut floating in outer space near Earth",
    "A medieval castle surrounded by autumn trees",
    "A tranquil Japanese zen garden with a stone path",
    "A bowl of fresh tropical fruit on a wooden table",
    "A lighthouse standing on rocky cliffs at dusk",
    "A snowy village with decorated Christmas trees",
]

# Models that produce images (not video); only these are supported for LPIPS eval.
_IMAGE_MODELS = {
    ModelType.FLUX_DEV,
    ModelType.FLUX_SCHNELL,
    ModelType.FLUX2_DEV,
    ModelType.SDXL_BASE,
    ModelType.SDXL_TURBO,
    ModelType.SD3_MEDIUM,
    ModelType.SD35_MEDIUM,
    ModelType.QWEN_IMAGE,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_to_tensor(img, device: str = "cuda") -> torch.Tensor:
    """Convert a PIL Image to an LPIPS-compatible [-1, 1] tensor."""
    t = TF.to_tensor(img).unsqueeze(0)   # [1, 3, H, W] in [0, 1]
    return (t * 2.0 - 1.0).to(device)


def _generate_image(pipe, model_type: ModelType, prompt: str, seed: int,
                    height: int, width: int, num_steps: int):
    """Run one deterministic inference pass and return the PIL image."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    # Carry over model-specific args (e.g. guidance_scale, max_sequence_length)
    # but override spatial dims so we always generate at the small LPIPS resolution.
    extra = {
        k: v
        for k, v in MODEL_DEFAULTS[model_type].get("inference_extra_args", {}).items()
        if k not in ("height", "width", "num_frames", "fps", "frame_rate")
    }
    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            generator=generator,
            **extra,
        )
    return output.images[0]


def _build_quant_config(fmt: QuantFormat, algo: QuantAlgo, trt_dtype: str) -> dict:
    """Assemble the mtq quant_cfg dict for the requested format + algorithm."""
    if fmt == QuantFormat.INT8:
        base = INT8_DEFAULT_CONFIG
    elif fmt == QuantFormat.FP8:
        base = FP8_DEFAULT_CONFIG
    else:  # FP4 / NVFP4
        base = NVFP4_DEFAULT_CONFIG

    cfg = copy.deepcopy(base)
    kwargs: dict = {}
    if algo == QuantAlgo.SMOOTHQUANT:
        kwargs["alpha"] = 1.0
    elif algo == QuantAlgo.SVDQUANT:
        kwargs["lowrank"] = 32
    set_quant_config_attr(cfg, trt_dtype, algo.value, **kwargs)
    return cfg


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_lpips_eval(
    model_type: ModelType,
    fmt: QuantFormat,
    algo: QuantAlgo,
    model_path: str | None,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    num_steps: int,
    threshold: float,
    calib_size: int,
    n_steps: int,
) -> float:
    """End-to-end LPIPS evaluation; returns the computed score."""
    try:
        import lpips as _lpips
    except ImportError:
        logger.error("lpips is not installed. Run: pip install lpips")
        sys.exit(2)

    # ------------------------------------------------------------------
    # 1. Load BF16 pipeline
    # ------------------------------------------------------------------
    backbone_name = MODEL_DEFAULTS[model_type].get("backbone", "transformer")
    model_cfg = ModelConfig(
        model_type=model_type,
        model_dtype={"default": torch.bfloat16},
        backbone=[backbone_name],
        override_model_path=Path(model_path) if model_path else None,
    )
    pm = PipelineManager(model_cfg, logger)
    pipe = pm.create_pipeline()
    pm.setup_device()

    # ------------------------------------------------------------------
    # 2. Generate BF16 golden image
    # ------------------------------------------------------------------
    logger.info("Generating BF16 golden image (seed=%d) ...", seed)
    golden_image = _generate_image(pipe, model_type, prompt, seed, height, width, num_steps)

    # ------------------------------------------------------------------
    # 3. Quantize backbone in-memory
    # ------------------------------------------------------------------
    logger.info("Quantizing: format=%s  algo=%s ...", fmt.value, algo.value)
    quant_cfg_dict = _build_quant_config(fmt, algo, trt_dtype="BFloat16")

    # Build calibration prompts (fixed list, no HF dataset download needed)
    calib_prompts = _CALIB_PROMPTS[:calib_size]
    extra_args = MODEL_DEFAULTS[model_type].get("inference_extra_args", {})
    # Strip video-specific keys
    extra_args = {
        k: v for k, v in extra_args.items()
        if k not in ("num_frames", "fps", "frame_rate")
    }

    def forward_loop(_backbone):
        for i, p in enumerate(calib_prompts):
            if i >= calib_size:
                break
            with torch.no_grad():
                pipe(
                    prompt=[p],
                    num_inference_steps=n_steps,
                    **extra_args,
                )

    for _name, backbone in pm.iter_backbones():
        mtq.quantize(backbone, quant_cfg_dict, forward_loop)

    # ------------------------------------------------------------------
    # 4. Generate quantized image (same seed)
    # ------------------------------------------------------------------
    logger.info("Generating quantized image (seed=%d) ...", seed)
    quant_image = _generate_image(pipe, model_type, prompt, seed, height, width, num_steps)

    # ------------------------------------------------------------------
    # 5. Compute LPIPS
    # ------------------------------------------------------------------
    logger.info("Computing LPIPS ...")
    lpips_fn = _lpips.LPIPS(net="alex").to("cuda")
    with torch.no_grad():
        score = lpips_fn(
            _image_to_tensor(golden_image),
            _image_to_tensor(quant_image),
        ).item()

    # Print in a parseable format so callers can grep the score if needed.
    print(f"LPIPS_SCORE: {score:.6f}")
    logger.info("LPIPS score: %.4f  (threshold: %.4f)", score, threshold)
    return score


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LPIPS quality evaluation for quantized diffusion models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, choices=[m.value for m in ModelType],
        help="Model identifier (must be an image-generation model)",
    )
    parser.add_argument(
        "--format", dest="fmt", default="fp8",
        choices=[f.value for f in QuantFormat],
        help="Quantization format",
    )
    parser.add_argument(
        "--quant-algo", default="max",
        choices=[a.value for a in QuantAlgo],
        help="Calibration algorithm",
    )
    parser.add_argument("--override-model-path", default=None,
                        help="Local model path; overrides the default HuggingFace model ID")
    parser.add_argument("--prompt", default=LPIPS_PROMPT,
                        help="Prompt used for both golden and quantized image generation")
    parser.add_argument("--seed", type=int, default=LPIPS_SEED,
                        help="RNG seed for deterministic generation")
    parser.add_argument("--height", type=int, default=LPIPS_HEIGHT,
                        help="Image height in pixels")
    parser.add_argument("--width", type=int, default=LPIPS_WIDTH,
                        help="Image width in pixels")
    parser.add_argument("--num-steps", type=int, default=LPIPS_NUM_STEPS,
                        help="Number of diffusion steps")
    parser.add_argument("--threshold", type=float, default=LPIPS_THRESHOLD,
                        help="LPIPS threshold; score above this value fails the test")
    parser.add_argument("--calib-size", type=int, default=LPIPS_CALIB_SIZE,
                        help="Number of calibration prompts")
    parser.add_argument("--n-steps", type=int, default=LPIPS_N_STEPS,
                        help="Number of diffusion steps per calibration sample")
    return parser.parse_args()


def main():
    args = _parse_args()
    model_type = ModelType(args.model)

    if model_type not in _IMAGE_MODELS:
        logger.error(
            "Model '%s' is not an image-generation model. "
            "LPIPS evaluation requires image output. Supported models: %s",
            args.model,
            ", ".join(m.value for m in _IMAGE_MODELS),
        )
        sys.exit(2)

    score = run_lpips_eval(
        model_type=model_type,
        fmt=QuantFormat(args.fmt),
        algo=QuantAlgo(args.quant_algo),
        model_path=args.override_model_path,
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_steps=args.num_steps,
        threshold=args.threshold,
        calib_size=args.calib_size,
        n_steps=args.n_steps,
    )

    if score > args.threshold:
        logger.error("FAIL — LPIPS %.4f exceeds threshold %.4f", score, args.threshold)
        sys.exit(1)

    logger.info("PASS — LPIPS %.4f <= threshold %.4f", score, args.threshold)


if __name__ == "__main__":
    main()
