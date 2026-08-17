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

"""Remote RTC inference engine.

Identical to :class:`RTCInferenceEngine` except that chunk production —
preprocessing, ``predict_action_chunk`` and postprocessing — happens on a
remote server (see ``lerobot.async_inference.rtc_server``) reached over
gRPC. Everything else (action queue, latency->delay compensation,
backpressure, warmup accounting, error backoff) is inherited unchanged:
the RPC round-trip is measured by the same clock as local compute, so
network latency feeds ``inference_delay`` automatically.

The transport is unauthenticated plaintext gRPC (this repo's transport
stack has no TLS/auth); only use it on a trusted network such as a
tailnet. Payloads are plain dicts through the weights-only codec in
``lerobot.transport.utils``.
"""

from __future__ import annotations

import logging
import time
from threading import Event

import grpc
import torch

from lerobot.transport.remote_rtc_spec import (
    PREDICT,
    READY,
    RESET,
    SETUP,
    WIRE_VERSION,
    method_path,
)
from lerobot.transport.utils import bytes_to_python_object, grpc_channel_options, python_object_to_bytes
from lerobot.utils.feature_utils import build_dataset_frame

from ..robot_wrapper import ThreadSafeRobot
from .rtc import RTCInferenceEngine, _normalize_prev_actions_length

logger = logging.getLogger(__name__)

# Cadence of Ready polls while waiting for the server to come up.
_READY_POLL_INTERVAL_S: float = 1.0
# Per-call deadline for the lightweight Ready/Reset RPCs.
_CONTROL_RPC_TIMEOUT_S: float = 5.0


class RemoteRTCInferenceEngine(RTCInferenceEngine):
    """RTC engine whose chunk production runs on a remote gRPC server."""

    def __init__(
        self,
        server_address: str,
        server_policy_type: str,
        server_policy_path: str,
        rename_map: dict[str, str],
        robot_wrapper: ThreadSafeRobot,
        rtc_config,
        hw_features: dict,
        ordered_action_keys: list[str],
        task: str,
        fps: float,
        rtc_queue_threshold: int = 30,
        shutdown_event: Event | None = None,
        connect_timeout_s: float = 20.0,
        predict_timeout_s: float = 10.0,
        warmup_inferences: int = 1,
    ) -> None:
        super().__init__(
            policy=None,
            preprocessor=None,
            postprocessor=None,
            robot_wrapper=robot_wrapper,
            rtc_config=rtc_config,
            hw_features=hw_features,
            task=task,
            fps=fps,
            device="cpu",  # client-side tensors stay on CPU; the server owns the accelerator
            use_torch_compile=False,
            rtc_queue_threshold=rtc_queue_threshold,
            shutdown_event=shutdown_event,
        )
        self._server_address = server_address
        self._server_policy_type = server_policy_type
        self._server_policy_path = server_policy_path
        self._rename_map = rename_map
        self._ordered_action_keys = ordered_action_keys
        self._connect_timeout_s = connect_timeout_s
        self._predict_timeout_s = predict_timeout_s

        # The first remote calls carry server-side cold start (CUDA kernels,
        # tokenizer); exclude them from the monotonic latency tracker.
        if warmup_inferences > 0:
            self._warmup_count = warmup_inferences
            self._compile_warmup_done.clear()

        self._channel: grpc.Channel | None = None
        self._methods: dict[str, grpc.UnaryUnaryMultiCallable] = {}
        self._logged_request_size = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect, run the Setup handshake, then start the RTC thread."""
        self._channel = grpc.insecure_channel(self._server_address, grpc_channel_options())
        self._methods = {
            name: self._channel.unary_unary(method_path(name)) for name in (SETUP, PREDICT, RESET, READY)
        }

        self._wait_for_server()
        self._setup_server()
        super().start()

    def stop(self) -> None:
        """Join the RTC thread first so no RPC is in flight, then close the channel."""
        super().stop()
        if self._channel is not None:
            self._channel.close()
            self._channel = None

    def _wait_for_server(self) -> None:
        deadline = time.perf_counter() + self._connect_timeout_s
        last_error: Exception | None = None
        while time.perf_counter() < deadline:
            try:
                resp = bytes_to_python_object(
                    self._methods[READY](python_object_to_bytes({}), timeout=_CONTROL_RPC_TIMEOUT_S)
                )
                logger.info(
                    "Remote RTC server at %s is ready (device=%s, policy_loaded=%s)",
                    self._server_address,
                    resp.get("device"),
                    resp.get("policy_loaded"),
                )
                return
            except grpc.RpcError as e:
                last_error = e
                time.sleep(_READY_POLL_INTERVAL_S)
        raise RuntimeError(
            f"Remote RTC server at {self._server_address} unreachable after "
            f"{self._connect_timeout_s:.0f}s: {last_error}"
        )

    def _setup_server(self) -> None:
        request = {
            "wire_version": WIRE_VERSION,
            "policy_type": self._server_policy_type,
            "pretrained_name_or_path": self._server_policy_path,
            "rename_map": self._rename_map,
            "task": self._task,
            "rtc": {
                "enabled": self._rtc_config.enabled,
                "execution_horizon": self._rtc_config.execution_horizon,
                "prefix_attention_schedule": self._rtc_config.prefix_attention_schedule.value,
                "max_guidance_weight": self._rtc_config.max_guidance_weight,
            },
        }
        try:
            resp = bytes_to_python_object(
                self._methods[SETUP](python_object_to_bytes(request), timeout=self._connect_timeout_s)
            )
        except grpc.RpcError as e:
            raise RuntimeError(f"Remote RTC Setup RPC failed: {e}") from e

        if not resp.get("ok"):
            raise RuntimeError(f"Remote RTC server rejected Setup: {resp.get('error')}")
        if resp.get("wire_version") != WIRE_VERSION:
            raise RuntimeError(
                f"Remote RTC wire version mismatch: client={WIRE_VERSION} server={resp.get('wire_version')} "
                "— both machines must run the same branch/commit."
            )
        action_dim = resp.get("action_dim")
        if action_dim != len(self._ordered_action_keys):
            raise RuntimeError(
                f"Remote policy action_dim={action_dim} does not match the robot's "
                f"{len(self._ordered_action_keys)} action keys ({self._ordered_action_keys})."
            )
        logger.info(
            "Remote RTC Setup complete: policy=%s chunk_size=%s action_dim=%s device=%s",
            self._server_policy_path,
            resp.get("chunk_size"),
            action_dim,
            resp.get("device"),
        )

    # ------------------------------------------------------------------
    # Chunk production (RTC thread)
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        obs: dict,
        inference_delay: int,
        prev_actions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_frame = build_dataset_frame(self._hw_features, obs, prefix="observation")

        if prev_actions is not None:
            prev_actions = _normalize_prev_actions_length(
                prev_actions, target_steps=self._rtc_config.execution_horizon
            ).cpu()

        request_bytes = python_object_to_bytes(
            {
                "obs_frame": obs_frame,
                "robot_type": self._robot.robot_type,
                "inference_delay": inference_delay,
                "prev_chunk_left_over": prev_actions,
            }
        )
        if not self._logged_request_size:
            self._logged_request_size = True
            logger.info(
                "Remote RTC Predict request size: %.2f MB (gRPC cap 4 MB — reduce camera "
                "resolution if RESOURCE_EXHAUSTED appears)",
                len(request_bytes) / 1e6,
            )

        try:
            resp = bytes_to_python_object(
                self._methods[PREDICT](request_bytes, timeout=self._predict_timeout_s)
            )
        except grpc.RpcError as e:
            # Re-raise into the base loop's error backoff; DEADLINE_EXCEEDED
            # here is what keeps a hung server from wedging the RTC thread.
            raise RuntimeError(f"Remote RTC Predict failed ({e.code() if hasattr(e, 'code') else e})") from e

        return resp["original"], resp["processed"]

    def _reset_inference_state(self) -> None:
        """Reset server-side policy/pipeline state; warn-and-continue on failure.

        A persistent outage surfaces via the Predict error-backoff path within
        seconds; crashing ``reset()`` between episodes would instead take down
        strategies that don't expect it.
        """
        if not self._methods:
            return
        try:
            self._methods[RESET](python_object_to_bytes({}), timeout=_CONTROL_RPC_TIMEOUT_S)
        except grpc.RpcError as e:
            logger.warning("Remote RTC Reset RPC failed (continuing): %s", e)
