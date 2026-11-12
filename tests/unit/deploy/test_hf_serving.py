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

"""CPU-only checks for serving recipe wiring and failure handling; no Hub access."""

import base64
import io
import json
import signal
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
from _test_utils import hf_serving as serving

RECIPES = Path(serving.__file__).parents[1] / "examples/hf_ptq/deploy_model_cards.json"


def _case(**kwargs):
    return serving.ServingCase(
        model_id="local/test-model",
        backend="vllm",
        gpu_count=1,
        min_sm=90,
        runtime="test",
        sources=[],
        **kwargs,
    )


def test_requested_models_have_explicit_unique_cases():
    cases = serving.load_serving_cases(RECIPES)
    expected = {
        "nvidia/DeepSeek-V4.1-Flash-1017-NVFP4": {"vllm", "sglang"},
        "nvidia/Muse-Spark-Open-816B-A42B-DFlash2": {"vllm", "sglang"},
        "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16": {"vllm", "sglang"},
        "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4": {"vllm", "sglang", "trtllm"},
        "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr": {
            "vllm",
            "sglang",
            "trtllm",
        },
        "stepfun-ai/Step-5-Preview-NVFP4": {"sglang"},
        "nvidia/Qwen-Image-NVFP4": {"trtllm"},
    }
    assert {case.model_id for case in cases} == expected.keys()
    for model_id, backends in expected.items():
        assert {case.backend for case in cases if case.model_id == model_id} == backends


@pytest.mark.parametrize(
    "case",
    [c for c in serving.load_serving_cases(RECIPES) if c.draft_model],
    ids=lambda c: c.test_id,
)
def test_draft_checkpoint_is_paired_with_target(case, tmp_path, monkeypatch):
    monkeypatch.setattr(serving, "_model_path", lambda model, download=False: "/models/" + model)
    command = serving._prepare_command(case, tmp_path, 12345)
    target_index = {"vllm": 2, "sglang": 4, "trtllm": 1}[case.backend]
    assert command[target_index] == "/models/" + case.target_model
    assert case.target_model != case.draft_model
    if case.config:
        config = json.loads((tmp_path / "server-config.yaml").read_text())
        assert config["speculative_config"]["speculative_model"] == "/models/" + case.draft_model
        assert case.config["speculative_config"]["speculative_model"] == "{draft}"
    else:
        assert any("/models/" + case.draft_model in arg for arg in command)
    assert "{draft}" not in " ".join(command)


def test_precision_and_image_routes_do_not_use_legacy_text_defaults(tmp_path, monkeypatch):
    cases = serving.load_serving_cases(RECIPES)
    for case in cases:
        if case.model_id.endswith("BF16"):
            assert "--quantization" not in case.args
        if "Step-5" in case.model_id:
            assert "megamoe" not in case.args
            assert case.min_sm == 100
        if "LiLiCorr" in case.model_id:
            assert case.gpu_name == "GB10"
    image_case = next(c for c in cases if c.image)
    monkeypatch.setattr(serving, "_model_path", lambda model, download=False: "/models/qwen-image")
    command = serving._prepare_command(image_case, tmp_path, 12345)
    assert command[:2] == ["trtllm-serve", "/models/qwen-image"]
    assert "--visual_gen_args" in command


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {"content": " "}}], "usage": {"completion_tokens": 5}},
        {"choices": [{"message": {"content": "hello"}}], "usage": {"completion_tokens": 0}},
    ],
)
def test_empty_or_non_generated_responses_fail(result, tmp_path):
    with pytest.raises(AssertionError):
        serving._check_response(_case(), result, tmp_path)


def test_reasoning_only_smoke_output_is_recorded(tmp_path):
    result = {
        "choices": [{"message": {"reasoning_content": "Thinking about a greeting"}}],
        "usage": {"completion_tokens": 4},
    }
    serving._check_response(_case(), result, tmp_path)
    assert json.loads((tmp_path / "response.json").read_text()) == result


def test_image_response_must_decode_as_expected_png(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    image_module.new("RGB", (512, 512), "red").save(buffer, format="PNG")
    result = {"data": [{"b64_json": base64.b64encode(buffer.getvalue()).decode()}]}
    serving._check_response(_case(image=True), result, tmp_path)
    assert (tmp_path / "generated.png").read_bytes() == buffer.getvalue()
    result["data"][0]["b64_json"] = base64.b64encode(b"not an image").decode()
    with pytest.raises(image_module.UnidentifiedImageError):
        serving._check_response(_case(image=True), result, tmp_path)


def test_server_exit_is_a_failure_not_a_skip():
    process = Mock(returncode=7)
    process.poll.return_value = 7
    with pytest.raises(RuntimeError, match="exited with code 7"):
        serving._wait_ready(process, "http://127.0.0.1:12345", 60)


def test_startup_timeout_retains_connection_error(monkeypatch):
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(serving, "_request_json", Mock(side_effect=OSError("connection refused")))
    monkeypatch.setattr(serving.time, "monotonic", Mock(side_effect=[0, 0, 2]))
    monkeypatch.setattr(serving.time, "sleep", Mock())
    with pytest.raises(TimeoutError, match="connection refused"):
        serving._wait_ready(process, "http://127.0.0.1:12345", 1)


@pytest.mark.parametrize(
    "failure", [RuntimeError("load failed"), TimeoutError("generation timed out")]
)
def test_server_group_is_cleaned_up_on_failure(failure, tmp_path, monkeypatch):
    process = Mock(pid=12345, returncode=None)
    process.poll.return_value = None
    monkeypatch.setattr(serving, "_check_environment", Mock())
    monkeypatch.setattr(serving.shutil, "which", Mock(return_value="/fake/vllm"))
    monkeypatch.setattr(
        serving, "_prepare_command", Mock(return_value=["/fake/vllm", "serve", "local/test-model"])
    )
    listener = Mock()
    listener.getsockname.return_value = ("127.0.0.1", 12345)
    context = Mock()
    context.__enter__ = Mock(return_value=listener)
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(serving.socket, "socket", Mock(return_value=context))
    monkeypatch.setattr(serving.subprocess, "Popen", Mock(return_value=process))
    kill = Mock()
    monkeypatch.setattr(serving.os, "killpg", kill)
    if isinstance(failure, RuntimeError):
        monkeypatch.setattr(serving, "_wait_ready", Mock(side_effect=failure))
    else:
        monkeypatch.setattr(serving, "_wait_ready", Mock(return_value="local/test-model"))
        monkeypatch.setattr(serving, "_request_json", Mock(side_effect=failure))
    with pytest.raises(type(failure), match=str(failure)):
        serving.run_serving_case(_case(), tmp_path)
    assert kill.call_args_list[0].args == (12345, signal.SIGTERM)
    assert kill.call_args_list[1].args == (12345, signal.SIGKILL)
    assert (tmp_path / "command.json").exists()


def test_cleanup_escalates_when_workers_do_not_exit(monkeypatch):
    process = Mock(pid=12345)
    process.wait.side_effect = [subprocess.TimeoutExpired("server", 20), 0]
    kill = Mock()
    monkeypatch.setattr(serving.os, "killpg", kill)
    serving._stop_process_group(process)
    assert kill.call_args_list[-1].args == (12345, signal.SIGKILL)


def test_unrelated_server_does_not_count_as_ready(monkeypatch):
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(
        serving, "_request_json", Mock(return_value={"data": [{"id": "another-model"}]})
    )
    monkeypatch.setattr(serving.time, "monotonic", Mock(side_effect=[0, 0, 2]))
    monkeypatch.setattr(serving.time, "sleep", Mock())
    with pytest.raises(TimeoutError, match="expected model: local/test-model"):
        serving._wait_ready(process, "http://127.0.0.1:12345", 1, expected_model="local/test-model")


def test_model_list_readiness_returns_expected_model(monkeypatch):
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(
        serving, "_request_json", Mock(return_value={"data": [{"id": "local/test-model"}]})
    )
    assert (
        serving._wait_ready(process, "http://127.0.0.1:12345", 1, "local/test-model")
        == "local/test-model"
    )
