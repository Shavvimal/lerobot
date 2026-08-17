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

"""Synchronous RTC inference server.

Serves ``predict_action_chunk`` (with full RTC kwargs) over gRPC for the
client-side :class:`lerobot.rollout.inference.remote_rtc.RemoteRTCInferenceEngine`.
Unlike ``policy_server.py`` (the async observation-queue stack), this server is
a plain request/response service: every ``Predict`` call runs preprocessing,
the policy, and postprocessing, and returns both the model-space and the
postprocessed chunk — the pair the client's RTC action queue needs.

Run on the GPU machine:

    python -m lerobot.async_inference.rtc_server \\
        --host=<tailscale-ip> --port=9032 --device=cuda \\
        --policy_path=outputs/train/<run>/checkpoints/<step>/pretrained_model

SECURITY: the transport is unauthenticated plaintext gRPC. Bind to a trusted
interface (e.g. a tailnet IP) only — anyone who can reach the port can load
checkpoints readable by this process and run inference. Payloads are decoded
with the weights-only unpickler from ``lerobot.transport.utils``.
"""

# NOTE: no `from __future__ import annotations` here — draccus.wrap() needs the
# config dataclass annotation resolved at decoration time, not as a string.

import logging
import signal
import threading
import traceback
from concurrent import futures
from dataclasses import dataclass
from pprint import pformat

import draccus
import grpc
import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCAttentionSchedule, RTCConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import RelativeActionsProcessorStep
from lerobot.transport.remote_rtc_spec import (
    METHOD_NAMES,
    PREDICT,
    READY,
    RESET,
    SERVICE_NAME,
    SETUP,
    WIRE_VERSION,
)
from lerobot.transport.utils import bytes_to_python_object, grpc_channel_options, python_object_to_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_POLICY_TYPES = {"pi05"}


@dataclass
class RTCServerConfig:
    # Bind address. Deliberately defaults to loopback — pass the Tailscale IP
    # explicitly to expose it; never bind 0.0.0.0 (no auth on this transport).
    host: str = "127.0.0.1"
    port: int = 9032
    device: str = "cuda"
    # Optional preload; the client's Setup RPC remains authoritative.
    policy_path: str | None = None
    policy_type: str = "pi05"


class RemoteRTCService:
    """Unary request/response handlers, registered via generic RPC handlers."""

    def __init__(self, config: RTCServerConfig) -> None:
        self.config = config
        self.device = config.device
        self.lock = threading.Lock()
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.task: str = ""
        self._loaded_signature: tuple | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(
        self,
        policy_type: str,
        pretrained_name_or_path: str,
        rtc: dict,
        rename_map: dict[str, str],
        task: str,
    ) -> None:
        if policy_type not in SUPPORTED_POLICY_TYPES:
            raise ValueError(f"Unsupported policy type {policy_type!r} (supported: {SUPPORTED_POLICY_TYPES})")

        signature = (
            policy_type,
            pretrained_name_or_path,
            tuple(sorted(rtc.items())),
            tuple(sorted(rename_map.items())),
        )
        if signature == self._loaded_signature and self.policy is not None:
            logger.info("Setup matches loaded policy — skipping reload, resetting state")
            self._reset_state()
            self.task = task
            return

        logger.info("Loading policy %s from '%s'...", policy_type, pretrained_name_or_path)
        policy_class = get_policy_class(policy_type)
        policy = policy_class.from_pretrained(pretrained_name_or_path)

        # RTC wiring must happen before device placement/eval — mirrors
        # rollout/context.py's build_rollout_context.
        policy.config.rtc_config = RTCConfig(
            enabled=rtc["enabled"],
            execution_horizon=rtc["execution_horizon"],
            prefix_attention_schedule=RTCAttentionSchedule(rtc["prefix_attention_schedule"]),
            max_guidance_weight=rtc["max_guidance_weight"],
        )
        if hasattr(policy, "init_rtc_processor"):
            policy.init_rtc_processor()

        policy = policy.to(self.device)
        policy.eval()

        device_override = {"device": self.device}
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": rename_map},
            },
            # Postprocessor pinned to CPU: responses are deserialized with a
            # weights-only torch.load that has no map_location, so tensors
            # must already be CPU when they hit the wire.
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

        if any(
            isinstance(step, RelativeActionsProcessorStep) and step.enabled
            for step in getattr(preprocessor, "steps", ())
        ):
            raise NotImplementedError("rtc_server does not support relative-action policies yet")

        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.task = task
        self._loaded_signature = signature
        logger.info(
            "Policy loaded on %s: chunk_size=%s task=%r rename_map=%s",
            self.device,
            getattr(policy.config, "chunk_size", "?"),
            task,
            rename_map,
        )

    def _reset_state(self) -> None:
        if self.policy is not None:
            self.policy.reset()
        if self.preprocessor is not None:
            self.preprocessor.reset()
        if self.postprocessor is not None:
            self.postprocessor.reset()

    @property
    def _action_dim(self) -> int | None:
        if self.policy is None:
            return None
        try:
            from lerobot.utils.constants import ACTION

            return self.policy.config.output_features[ACTION].shape[0]
        except Exception:
            names = getattr(self.policy.config, "action_feature_names", None)
            return len(names) if names else None

    # ------------------------------------------------------------------
    # RPC handlers (each: bytes request -> bytes response)
    # ------------------------------------------------------------------

    def ready(self, request: bytes, context: grpc.ServicerContext) -> bytes:
        return python_object_to_bytes(
            {"ok": True, "policy_loaded": self.policy is not None, "device": self.device}
        )

    def setup(self, request: bytes, context: grpc.ServicerContext) -> bytes:
        req = bytes_to_python_object(request)
        if req.get("wire_version") != WIRE_VERSION:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"wire_version mismatch: server={WIRE_VERSION} client={req.get('wire_version')}",
            )
        try:
            with self.lock:
                self._load(
                    policy_type=req["policy_type"],
                    pretrained_name_or_path=req["pretrained_name_or_path"],
                    rtc=req["rtc"],
                    rename_map=req.get("rename_map") or {},
                    task=req.get("task", ""),
                )
        except Exception as e:
            logger.error("Setup failed:\n%s", traceback.format_exc())
            return python_object_to_bytes({"ok": False, "error": str(e), "wire_version": WIRE_VERSION})

        return python_object_to_bytes(
            {
                "ok": True,
                "error": None,
                "wire_version": WIRE_VERSION,
                "action_dim": self._action_dim,
                "chunk_size": getattr(self.policy.config, "chunk_size", None),
                "device": self.device,
            }
        )

    def predict(self, request: bytes, context: grpc.ServicerContext) -> bytes:
        if self.policy is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "No policy loaded — call Setup first")
        try:
            req = bytes_to_python_object(request)
            with self.lock:
                obs = prepare_observation_for_inference(
                    req["obs_frame"], torch.device(self.device), self.task, req.get("robot_type")
                )
                # Tokenizer parity with the local RTC engine (rtc.py):
                # the pi05 tokenizer step expects task as a list.
                obs["task"] = [self.task]
                batch = self.preprocessor(obs)

                prev = req.get("prev_chunk_left_over")
                if prev is not None:
                    prev = prev.to(self.device)

                # No torch.inference_mode() wrapper: predict_action_chunk is
                # already @torch.no_grad(), and RTC guidance re-enables grad
                # internally (inference-mode tensors would break autograd).
                actions = self.policy.predict_action_chunk(
                    batch,
                    inference_delay=req.get("inference_delay", 0),
                    prev_chunk_left_over=prev,
                )

                original = actions.squeeze(0).detach().clone().cpu()
                processed = self.postprocessor(actions).squeeze(0)
            return python_object_to_bytes({"original": original, "processed": processed})
        except Exception as e:
            logger.error("Predict failed:\n%s", traceback.format_exc())
            context.abort(grpc.StatusCode.INTERNAL, f"Predict failed: {e}")

    def reset(self, request: bytes, context: grpc.ServicerContext) -> bytes:
        with self.lock:
            self._reset_state()
        return python_object_to_bytes({"ok": True})


def _build_server(service: RemoteRTCService) -> grpc.Server:
    handlers = {
        SETUP: grpc.unary_unary_rpc_method_handler(service.setup),
        PREDICT: grpc.unary_unary_rpc_method_handler(service.predict),
        RESET: grpc.unary_unary_rpc_method_handler(service.reset),
        READY: grpc.unary_unary_rpc_method_handler(service.ready),
    }
    assert set(handlers) == set(METHOD_NAMES)
    # NOTE: options= on the server matters — without it gRPC's defaults cap
    # receive size at 4MB with no keepalive tuning (policy_server.py omits
    # this; don't copy that).
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=grpc_channel_options())
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(SERVICE_NAME, handlers),))
    return server


@draccus.wrap()
def serve(cfg: RTCServerConfig) -> None:
    logger.info(pformat(cfg))
    service = RemoteRTCService(cfg)

    if cfg.policy_path:
        logger.info("Preloading policy (Setup RPC remains authoritative)...")
        service._load(
            policy_type=cfg.policy_type,
            pretrained_name_or_path=cfg.policy_path,
            rtc={
                "enabled": True,
                "execution_horizon": RTCConfig().execution_horizon,
                "prefix_attention_schedule": RTCConfig().prefix_attention_schedule.value,
                "max_guidance_weight": RTCConfig().max_guidance_weight,
            },
            rename_map={},
            task="",
        )

    server = _build_server(service)
    bound = server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    server.start()
    logger.info("RemoteRTC server listening on %s:%d (device=%s)", cfg.host, bound, cfg.device)

    stop_event = threading.Event()

    def _graceful_stop(signum, frame):
        logger.info("Signal %s received, stopping server...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _graceful_stop)
    signal.signal(signal.SIGTERM, _graceful_stop)

    stop_event.wait()
    server.stop(grace=5).wait()
    logger.info("Server stopped")


if __name__ == "__main__":
    serve()
