# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-process end-to-end tests for the remote RTC inference engine."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("grpc")

from lerobot.async_inference.rtc_server import RemoteRTCService, RTCServerConfig, _build_server
from lerobot.rollout.inference.remote_rtc import RemoteRTCInferenceEngine

CHUNK_SIZE = 20
ACTION_DIM = 6
ACTION_KEYS = [f"j{i}.pos" for i in range(ACTION_DIM)]
HW_FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (ACTION_DIM,),
        "names": ACTION_KEYS,
    }
}
OBS = {k: float(i) for i, k in enumerate(ACTION_KEYS)}


class MockPolicy:
    """Records RTC kwargs; returns a constant chunk."""

    def __init__(self):
        self.calls: list[dict] = []
        self.config = MagicMock()
        self.config.chunk_size = CHUNK_SIZE
        self.config.action_feature_names = ACTION_KEYS
        # Force the server's _action_dim fallback path (len(action_feature_names)).
        self.config.output_features = {}

    def predict_action_chunk(self, batch, inference_delay=0, prev_chunk_left_over=None):
        self.calls.append({"inference_delay": inference_delay, "prev_chunk_left_over": prev_chunk_left_over})
        return torch.zeros(1, CHUNK_SIZE, ACTION_DIM)

    def reset(self):
        pass

    def to(self, device):
        return self

    def eval(self):
        return self


class IdentityPipeline:
    def __call__(self, x):
        return x

    def reset(self):
        pass

    steps = ()


@pytest.fixture
def server_and_service():
    service = RemoteRTCService(RTCServerConfig(device="cpu"))

    def fake_load(policy_type, pretrained_name_or_path, rtc, rename_map, task):
        service.policy = MockPolicy()
        service.preprocessor = IdentityPipeline()
        service.postprocessor = IdentityPipeline()
        service.task = task

    service._load = fake_load
    server = _build_server(service)
    port = server.add_insecure_port("localhost:0")
    server.start()
    yield f"localhost:{port}", service
    server.stop(grace=None)


def make_engine(address, **kwargs):
    robot_wrapper = MagicMock()
    robot_wrapper.robot_type = "so101_follower"
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    defaults = {
        "server_address": address,
        "server_policy_type": "pi05",
        "server_policy_path": "/fake/path",
        "rename_map": {},
        "robot_wrapper": robot_wrapper,
        "rtc_config": RTCConfig(),
        "hw_features": HW_FEATURES,
        "ordered_action_keys": ACTION_KEYS,
        "task": "test task",
        "fps": 30.0,
        "connect_timeout_s": 5.0,
        "predict_timeout_s": 5.0,
    }
    defaults.update(kwargs)
    return RemoteRTCInferenceEngine(**defaults)


def _poll_action(engine, timeout=10.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        action = engine.get_action(None)
        if action is not None:
            return action
        time.sleep(0.01)
    raise TimeoutError("no action produced")


def test_engine_lifecycle_and_prev_chunk_roundtrip(server_and_service):
    address, service = server_and_service
    engine = make_engine(address)
    try:
        assert not engine.ready  # warmup_inferences=1 gates readiness on the first chunk
        engine.start()
        engine.notify_observation(OBS)
        engine.resume()

        action = _poll_action(engine)
        assert action.shape == (ACTION_DIM,)

        deadline = time.perf_counter() + 10.0
        while len(service.policy.calls) < 2 and time.perf_counter() < deadline:
            engine.notify_observation(OBS)
            engine.get_action(None)  # consume so leftovers shrink and refills trigger
            time.sleep(0.01)
        assert len(service.policy.calls) >= 2

        assert engine.ready  # first round-trip completed warmup
        assert not engine.failed

        first, second = service.policy.calls[0], service.policy.calls[1]
        assert first["prev_chunk_left_over"] is None
        prev = second["prev_chunk_left_over"]
        assert prev is not None
        # Client normalizes the leftover tail to exactly execution_horizon steps.
        assert prev.shape == (engine._rtc_config.execution_horizon, ACTION_DIM)

        engine.reset()  # exercises the Reset RPC (warn-and-continue semantics)
    finally:
        engine.stop()
    assert engine._channel is None


def test_setup_failure_fails_fast(server_and_service):
    address, service = server_and_service

    def failing_load(**kwargs):
        raise ValueError("bad checkpoint")

    service._load = lambda **kw: (_ for _ in ()).throw(ValueError("bad checkpoint"))
    engine = make_engine(address)
    with pytest.raises(RuntimeError, match="bad checkpoint"):
        engine.start()
    engine.stop()


def test_action_dim_mismatch_fails_fast(server_and_service):
    address, _ = server_and_service
    engine = make_engine(address, ordered_action_keys=[f"j{i}.pos" for i in range(7)])
    with pytest.raises(RuntimeError, match="action_dim"):
        engine.start()
    engine.stop()


def test_predict_errors_trip_failure_and_shutdown(server_and_service, monkeypatch):
    address, _ = server_and_service
    from threading import Event

    import lerobot.rollout.inference.rtc as rtc_mod

    monkeypatch.setattr(rtc_mod, "_RTC_MAX_CONSECUTIVE_ERRORS", 2)
    monkeypatch.setattr(rtc_mod, "_RTC_ERROR_RETRY_DELAY_S", 0.01)

    shutdown = Event()
    engine = make_engine(address, shutdown_event=shutdown)
    try:
        engine.start()

        def broken_predict(*args, **kwargs):
            raise RuntimeError("server exploded")

        engine._methods = dict(engine._methods)
        from lerobot.transport.remote_rtc_spec import PREDICT

        engine._methods[PREDICT] = broken_predict

        engine.notify_observation(OBS)
        engine.resume()

        deadline = time.perf_counter() + 10.0
        while not engine.failed and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert engine.failed
        assert shutdown.is_set()
    finally:
        engine.stop()
