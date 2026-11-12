# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Run model-card serving recipes without changing the legacy offline deployer."""

import base64
import copy
import importlib.util
import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

import pytest

__all__ = ["ServingCase", "load_serving_cases", "run_serving_case"]


@dataclass(frozen=True)
class ServingCase:
    """One checkpoint/runtime recipe and its documented hardware requirements."""

    model_id: str
    backend: str
    gpu_count: int
    min_sm: int
    runtime: str
    sources: list[str]
    args: list[str] = field(default_factory=list)
    target_model: str = ""
    draft_model: str = ""
    gpu_name: str = ""
    config: dict = field(default_factory=dict)
    config_flag: str = "--config"
    environment: dict = field(default_factory=dict)
    image: bool = False

    @property
    def test_id(self):
        return f"{self.model_id}_{self.backend}"


def load_serving_cases(path):
    """Load explicit cases; missing runtimes are skipped at execution, not collection."""
    cases = [ServingCase(**entry) for entry in json.loads(Path(path).read_text())]
    ids = [case.test_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate model/backend serving case")
    for case in cases:
        if case.backend not in ("vllm", "sglang", "trtllm") or case.gpu_count < 1:
            raise ValueError(f"Invalid backend/GPU count: {case.test_id}")
        if case.draft_model and not case.target_model:
            raise ValueError(f"Draft model requires a target: {case.test_id}")
    return cases


def _check_environment(case):
    # CUDA is needed only for execution, so CPU-only recipe tests can import this module.
    import torch

    if importlib.util.find_spec({"trtllm": "tensorrt_llm"}.get(case.backend, case.backend)) is None:
        pytest.skip(f"{case.backend} unavailable; required runtime: {case.runtime}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < case.gpu_count:
        pytest.skip(f"Requires {case.gpu_count} visible CUDA GPUs; recipe: {case.runtime}")
    for index in range(case.gpu_count):
        major, minor = torch.cuda.get_device_capability(index)
        if major * 10 + minor < case.min_sm:
            pytest.skip(f"Requires SM{case.min_sm} or newer")
        if case.gpu_name and case.gpu_name not in torch.cuda.get_device_name(index):
            pytest.skip(f"This recipe is specific to {case.gpu_name}")


def _model_path(model_id, download=False):
    local_root = os.environ.get("MODELOPT_LOCAL_MODEL_ROOT")
    if local_root:
        path = Path(local_root) / model_id
        if not path.is_dir():
            raise FileNotFoundError(f"Missing local checkpoint: {path}")
        return str(path.resolve())
    if download:
        # Only TRT-LLM recipes require a local snapshot; never download during collection.
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=model_id)
    return model_id


def _prepare_command(case, work_dir, port):
    target = _model_path(case.target_model or case.model_id, download=case.backend == "trtllm")
    draft = (
        _model_path(case.draft_model, download=case.backend == "trtllm") if case.draft_model else ""
    )
    args = [arg.replace("{draft}", draft) for arg in case.args]
    if case.backend == "vllm":
        command = ["vllm", "serve", target, "--tensor-parallel-size", str(case.gpu_count)]
    elif case.backend == "sglang":
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            target,
            "--tp-size",
            str(case.gpu_count),
        ]
    else:
        command = ["trtllm-serve", target]
    if case.config:
        config = copy.deepcopy(case.config)
        if draft:
            config["speculative_config"]["speculative_model"] = draft
        config_path = work_dir / "server-config.yaml"
        # JSON is also valid YAML; keep the exact numeric/boolean recipe values.
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        command.extend([case.config_flag, str(config_path)])
    return [*command, *args, "--host", "127.0.0.1", "--port", str(port)]


def _request_json(base_url, path, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        base_url + path, data=data, headers={"Content-Type": "application/json"}
    )
    # Local test traffic must not use a corporate HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def _wait_ready(process, base_url, timeout, expected_model=None):
    deadline = time.monotonic() + timeout
    last_error = "not ready"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Serving process exited with code {process.returncode}")
        try:
            models = _request_json(base_url, "/v1/models")
            for model in models.get("data", []):
                model_id = model.get("id")
                if model_id and (expected_model is None or model_id == expected_model):
                    if process.poll() is not None:
                        raise RuntimeError(f"Serving process exited with code {process.returncode}")
                    return model_id
            last_error = f"Endpoint did not advertise the expected model: {expected_model}"
        except (OSError, ValueError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"Server startup exceeded {timeout}s: {last_error}")


def _stop_process_group(process):
    # Stop descendants too, including workers left behind by a failed server parent.
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _check_response(case, result, work_dir):
    (work_dir / "response.json").write_text(json.dumps(result, indent=2) + "\n")
    if case.image:
        # Pillow is only required by the VisualGen image case.
        from PIL import Image

        assert len(result.get("data", [])) == 1, "Expected one generated image"
        raw = base64.b64decode(result["data"][0]["b64_json"], validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            assert image.format == "PNG" and image.size == (512, 512)
            image.verify()
        (work_dir / "generated.png").write_bytes(raw)
    else:
        choices = result.get("choices", [])
        assert len(choices) == 1, "Expected one generated response"
        message = choices[0].get("message", {})
        text = (
            message.get("content") or message.get("reasoning_content") or message.get("reasoning")
        )
        assert isinstance(text, str) and text.strip(), "Generated text/reasoning is empty"
        assert result.get("usage", {}).get("completion_tokens", 0) > 0, "No generated tokens"
        print(f"Generated text: {text!r}")


def run_serving_case(case, work_dir):
    """Start a loopback-only server, check real generation, and reap its workers."""
    _check_environment(case)
    executable = {"vllm": "vllm", "sglang": sys.executable, "trtllm": "trtllm-serve"}[case.backend]
    if not shutil.which(executable):
        pytest.skip(f"Missing executable {executable}; required runtime: {case.runtime}")
    work_dir = Path(work_dir).resolve()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = _prepare_command(case, work_dir, port)
    environment = {**os.environ, **case.environment}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if visible == [""]:
        visible = [str(i) for i in range(case.gpu_count)]
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(visible[: case.gpu_count])
    (work_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    print(f"Deploying model: {case.model_id} with backend: {case.backend}")
    print(f"Target model: {case.target_model or case.model_id}; required runtime: {case.runtime}")
    print(f"Serving artifacts: {work_dir}")
    base_url = f"http://127.0.0.1:{port}"
    log_path = work_dir / "server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True
        )
        try:
            target_index = {"vllm": 2, "sglang": 4, "trtllm": 1}[case.backend]
            model_name = _wait_ready(
                process, base_url, timeout=3600, expected_model=command[target_index]
            )
            if case.image:
                payload = {
                    "model": model_name,
                    "prompt": "A red ceramic teapot on a wooden table.",
                    "negative_prompt": " ",
                    "size": "512x512",
                    "num_inference_steps": 50,
                    "guidance_scale": 4.0,
                    "seed": 42,
                    "response_format": "b64_json",
                    "format": "png",
                }
                path = "/v1/images/generations"
            else:
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
                    "max_tokens": 128,
                    "temperature": 1.0,
                    "top_p": 0.95,
                }
                path = "/v1/chat/completions"
            _check_response(case, _request_json(base_url, path, payload, timeout=600), work_dir)
        finally:
            try:
                _stop_process_group(process)
            finally:
                log.flush()
                with log_path.open("rb") as saved_log:
                    saved_log.seek(max(0, log_path.stat().st_size - 65536))
                    print(saved_log.read().decode(errors="replace"))
