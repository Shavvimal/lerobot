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

"""Inference engine configs and factory.

Selection is explicit via ``--inference.type=sync|rtc``.  Adding a new
backend requires registering its config subclass and dispatching it in
:func:`create_inference_engine`.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from threading import Event

import draccus

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import PolicyProcessorPipeline

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine
from .rtc import RTCInferenceEngine
from .sync import SyncInferenceEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass
class InferenceEngineConfig(draccus.ChoiceRegistry, abc.ABC):
    """Abstract base for inference backend configuration.

    Use ``--inference.type=<name>`` on the CLI to select a backend.
    """

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@InferenceEngineConfig.register_subclass("sync")
@dataclass
class SyncInferenceConfig(InferenceEngineConfig):
    """Inline synchronous inference (one policy call per control tick)."""


@InferenceEngineConfig.register_subclass("rtc")
@dataclass
class RTCInferenceConfig(InferenceEngineConfig):
    """Real-Time Chunking: async policy inference in a background thread."""

    # Eagerly constructed so draccus exposes nested fields directly on the CLI
    # (e.g. ``--inference.rtc.execution_horizon=...``).
    rtc: RTCConfig = field(default_factory=RTCConfig)
    queue_threshold: int = 30


@InferenceEngineConfig.register_subclass("remote_rtc")
@dataclass
class RemoteRTCInferenceConfig(RTCInferenceConfig):
    """RTC with the policy served on a remote machine over gRPC.

    All RTC machinery (action queue, latency->delay compensation,
    execution horizon, backpressure) runs locally; only preprocessing +
    ``predict_action_chunk`` + postprocessing execute on the server.
    Subclasses :class:`RTCInferenceConfig` so ``--inference.rtc.*`` flags
    and the rollout context's RTC checks apply unchanged.
    """

    # "host:port" of the remote RTC server (see lerobot.async_inference.rtc_server).
    server_address: str = ""
    # Checkpoint path/hub-id AS SEEN BY THE SERVER. Defaults to the client's
    # --policy.path, which is only correct when both machines share the path.
    server_policy_path: str | None = None
    connect_timeout_s: float = 20.0
    predict_timeout_s: float = 10.0
    # First remote calls carry CUDA/tokenizer cold-start; exclude them from the
    # (monotonic) latency tracker so delay compensation reflects steady state.
    warmup_inferences: int = 1

    def __post_init__(self) -> None:
        if not self.server_address:
            raise ValueError("remote_rtc requires --inference.server_address=<host:port>")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_inference_engine(
    config: InferenceEngineConfig,
    *,
    policy: PreTrainedPolicy | None,
    preprocessor: PolicyProcessorPipeline | None,
    postprocessor: PolicyProcessorPipeline | None,
    robot_wrapper: ThreadSafeRobot,
    hw_features: dict,
    dataset_features: dict,
    ordered_action_keys: list[str],
    task: str,
    fps: float,
    device: str | None,
    use_torch_compile: bool = False,
    compile_warmup_inferences: int = 2,
    shutdown_event: Event | None = None,
    policy_type: str | None = None,
    pretrained_path: str | None = None,
    rename_map: dict[str, str] | None = None,
) -> InferenceEngine:
    """Instantiate the appropriate inference engine from a config object."""
    logger.info("Creating inference engine: %s", config.type)
    # NOTE: RemoteRTCInferenceConfig subclasses RTCInferenceConfig, so this
    # branch MUST come before the isinstance(config, RTCInferenceConfig) check
    # below or the remote config would silently build a local RTC engine.
    if isinstance(config, RemoteRTCInferenceConfig):
        from .remote_rtc import RemoteRTCInferenceEngine

        return RemoteRTCInferenceEngine(
            server_address=config.server_address,
            server_policy_type=policy_type or "",
            server_policy_path=config.server_policy_path or pretrained_path or "",
            rename_map=rename_map or {},
            robot_wrapper=robot_wrapper,
            rtc_config=config.rtc,
            hw_features=hw_features,
            ordered_action_keys=ordered_action_keys,
            task=task,
            fps=fps,
            rtc_queue_threshold=config.queue_threshold,
            shutdown_event=shutdown_event,
            connect_timeout_s=config.connect_timeout_s,
            predict_timeout_s=config.predict_timeout_s,
            warmup_inferences=config.warmup_inferences,
        )
    if isinstance(config, SyncInferenceConfig):
        return SyncInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_features=dataset_features,
            ordered_action_keys=ordered_action_keys,
            task=task,
            device=device,
            robot_type=robot_wrapper.robot_type,
        )
    if isinstance(config, RTCInferenceConfig):
        return RTCInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robot_wrapper=robot_wrapper,
            rtc_config=config.rtc,
            hw_features=hw_features,
            task=task,
            fps=fps,
            device=device,
            use_torch_compile=use_torch_compile,
            compile_warmup_inferences=compile_warmup_inferences,
            rtc_queue_threshold=config.queue_threshold,
            shutdown_event=shutdown_event,
        )
    raise ValueError(f"Unknown inference engine type: {type(config).__name__}")
