#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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

import functools
import io
import json
import logging
from multiprocessing.synchronize import Event as MpEvent
from queue import Queue
from typing import Any

import torch

from lerobot.utils.transition import Transition

from . import services_pb2

# FIX for protobuf: Assign the enum to a variable and ignore the type error once
TransferState = services_pb2.TransferState  # type: ignore[attr-defined]

CHUNK_SIZE = 2 * 1024 * 1024  # 2 MB
MAX_MESSAGE_SIZE = 4 * 1024 * 1024  # 4 MB — gRPC's per-frame limit
# Cap on a *reassembled* message. `MAX_MESSAGE_SIZE` bounds each individual
# gRPC frame, but `receive_bytes_in_chunks` concatenates an unbounded number of
# them, so only this stops an unauthenticated peer from streaming chunks until
# the host runs out of memory. Far above any legitimate observation payload.
MAX_REASSEMBLED_MESSAGE_BYTES = 256 * 1024 * 1024  # 256 MB
# Model weights are the one payload class that legitimately exceeds the general
# cap: 256 MB is only ~67M fp32 parameters, so the learner -> actor parameter
# push would start failing at runtime on larger policies. Still bounded, so an
# unbounded stream is refused on that path too.
MAX_PARAMETERS_MESSAGE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


def bytes_buffer_size(buffer: io.BytesIO) -> int:
    buffer.seek(0, io.SEEK_END)
    result = buffer.tell()
    buffer.seek(0)
    return result


def send_bytes_in_chunks(buffer: bytes, message_class: Any, log_prefix: str = "", silent: bool = True):
    bytes_buffer: io.BytesIO = io.BytesIO(buffer)
    size_in_bytes = bytes_buffer_size(bytes_buffer)

    sent_bytes = 0

    logging_method = logging.info if not silent else logging.debug

    logging_method(f"{log_prefix} Buffer size {size_in_bytes / 1024 / 1024} MB with")

    while sent_bytes < size_in_bytes:
        transfer_state = TransferState.TRANSFER_MIDDLE

        if sent_bytes + CHUNK_SIZE >= size_in_bytes:
            transfer_state = TransferState.TRANSFER_END
        elif sent_bytes == 0:
            transfer_state = TransferState.TRANSFER_BEGIN

        size_to_read = min(CHUNK_SIZE, size_in_bytes - sent_bytes)
        chunk = bytes_buffer.read(size_to_read)

        yield message_class(transfer_state=transfer_state, data=chunk)
        sent_bytes += size_to_read
        logging_method(f"{log_prefix} Sent {sent_bytes}/{size_in_bytes} bytes with state {transfer_state}")

    logging_method(f"{log_prefix} Published {sent_bytes / 1024 / 1024} MB")


def receive_bytes_in_chunks(
    iterator,
    queue: Queue | None,
    shutdown_event: MpEvent,
    log_prefix: str = "",
    *,
    max_message_bytes: int = MAX_REASSEMBLED_MESSAGE_BYTES,
):
    """Reassemble a chunked gRPC stream into a single message.

    Args:
        max_message_bytes: Hard cap on one reassembled message. The stream is
            unauthenticated, so without a cap a peer that never sends
            ``TRANSFER_END`` grows this buffer until the host runs out of
            memory — on a machine that is driving a robot.

    Raises:
        ValueError: if a message exceeds ``max_message_bytes``.
    """
    bytes_buffer = io.BytesIO()
    step = 0

    def _guard_size() -> None:
        # `tell()` rather than `bytes_buffer_size()`: the latter rewinds the
        # buffer to position 0, which would make the next write overwrite what
        # has already been reassembled. Called straight after a write, the
        # position is the number of bytes accumulated.
        size = bytes_buffer.tell()
        if size > max_message_bytes:
            raise ValueError(
                f"{log_prefix} Message exceeds {max_message_bytes} bytes (got at least {size}); "
                f"aborting reassembly."
            )

    logging.info(f"{log_prefix} Starting receiver")
    for item in iterator:
        logging.debug(f"{log_prefix} Received item")
        if shutdown_event.is_set():
            logging.info(f"{log_prefix} Shutting down receiver")
            return

        if item.transfer_state == TransferState.TRANSFER_BEGIN:
            bytes_buffer.seek(0)
            bytes_buffer.truncate(0)
            bytes_buffer.write(item.data)
            _guard_size()
            logging.debug(f"{log_prefix} Received data at step 0")
            step = 0
        elif item.transfer_state == TransferState.TRANSFER_MIDDLE:
            bytes_buffer.write(item.data)
            _guard_size()
            step += 1
            logging.debug(f"{log_prefix} Received data at step {step}")
        elif item.transfer_state == TransferState.TRANSFER_END:
            bytes_buffer.write(item.data)
            logging.debug(f"{log_prefix} Received data at step end size {bytes_buffer_size(bytes_buffer)}")

            if queue is not None:
                queue.put(bytes_buffer.getvalue())
            else:
                return bytes_buffer.getvalue()

            bytes_buffer.seek(0)
            bytes_buffer.truncate(0)
            step = 0

            logging.debug(f"{log_prefix} Queue updated")
        else:
            logging.warning(f"{log_prefix} Received unknown transfer state {item.transfer_state}")
            raise ValueError(f"Received unknown transfer state {item.transfer_state}")


def state_to_bytes(state_dict: dict[str, torch.Tensor]) -> bytes:
    """Convert model state dict to flat array for transmission"""
    bytes_buffer = io.BytesIO()

    torch.save(state_dict, bytes_buffer)

    return bytes_buffer.getvalue()


def bytes_to_state_dict(buffer: bytes) -> dict[str, torch.Tensor]:
    bytes_buffer = io.BytesIO(buffer)
    bytes_buffer.seek(0)
    return torch.load(bytes_buffer, weights_only=True)


@functools.lru_cache(maxsize=1)
def _wire_safe_globals() -> list[Any]:
    """Types allowed to be reconstructed from a gRPC payload.

    Everything crossing the wire is either a plain scalar/container (handled by
    the weights-only unpickler natively), a tensor, a numpy array, or one of a
    small set of lerobot dataclasses. Anything else — in particular any callable
    an attacker could name in a ``__reduce__`` — must be refused.

    Imports are deferred: ``lerobot.async_inference`` imports this module, so
    resolving these at module scope would be circular.
    """
    import numpy as np

    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

    allowed: list[Any] = [FeatureType, NormalizationMode, PolicyFeature, np.ndarray, np.dtype]

    # `lerobot[async]` is optional; the RL path uses this codec without it.
    try:
        from lerobot.async_inference.helpers import (
            RemotePolicyConfig,
            TimedAction,
            TimedData,
            TimedObservation,
        )

        allowed += [TimedData, TimedAction, TimedObservation, RemotePolicyConfig]
    except ImportError:
        pass

    # numpy's array-reconstruction hooks. The module moved in numpy 2
    # (`numpy.core` -> `numpy._core`), so take whichever is present.
    for mod_name in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try:
            mod = __import__(mod_name, fromlist=["_reconstruct"])
        except ImportError:
            continue
        allowed += [fn for fn in (getattr(mod, a, None) for a in ("_reconstruct", "scalar")) if fn]

    allowed += [
        getattr(np, name)
        for name in ("bool_", "int8", "int16", "int32", "int64", "uint8", "float16", "float32", "float64")
        if hasattr(np, name)
    ]

    # numpy >= 2 stores an array's dtype as a concrete class from `numpy.dtypes`
    # (e.g. `numpy.dtypes.UInt8DType`). These are inert type objects.
    try:
        import numpy.dtypes as np_dtypes

        allowed += [getattr(np_dtypes, n) for n in dir(np_dtypes) if n.endswith("DType")]
    except ImportError:
        pass

    return allowed


@functools.lru_cache(maxsize=1)
def _register_wire_safe_globals() -> None:
    """Add the wire allowlist to torch's weights-only allowlist, once per process.

    Deliberately *not* ``torch.serialization.safe_globals(...)`` as a context
    manager. That manager unions the allowlist into a module-global set on entry
    and set-differences it out on exit, with no reference counting, so two
    concurrent deserializations race: the first to exit strips the allowlist out
    from under the second, which then fails with ``UnpicklingError``. The policy
    server runs gRPC on four threads and deserializes on the observation hot
    path, so that race drops observations on a robot control loop — and a second
    connection is enough for an unauthenticated peer to induce it deliberately.

    Registering once and never removing is race-free, and safe to leave in place:
    every entry is an inert type object or numpy's array-reconstruction hook,
    none of which grant code execution to any other ``torch.load`` in the
    process. Re-entry is harmless anyway — the underlying add is a set union.
    """
    torch.serialization.add_safe_globals(_wire_safe_globals())


def python_object_to_bytes(python_object: Any) -> bytes:
    bytes_buffer = io.BytesIO()
    torch.save(python_object, bytes_buffer)
    return bytes_buffer.getvalue()


def bytes_to_python_object(buffer: bytes) -> Any:
    """Deserialize a gRPC payload without granting it code execution.

    The transport is unauthenticated plaintext gRPC, so these bytes are
    attacker-controlled by anyone who can reach the port. A bare
    ``pickle.load`` here executes whatever the sender names in ``__reduce__``;
    the weights-only unpickler plus an explicit allowlist does not.

    Raises:
        pickle.UnpicklingError: if the payload references a non-allowlisted type.
    """
    _register_wire_safe_globals()
    bytes_buffer = io.BytesIO(buffer)
    bytes_buffer.seek(0)
    return torch.load(bytes_buffer, weights_only=True)


def bytes_to_transitions(buffer: bytes) -> list[Transition]:
    bytes_buffer = io.BytesIO(buffer)
    bytes_buffer.seek(0)
    transitions = torch.load(bytes_buffer, weights_only=True)
    return transitions


def transitions_to_bytes(transitions: list[Transition]) -> bytes:
    bytes_buffer = io.BytesIO()
    torch.save(transitions, bytes_buffer)
    return bytes_buffer.getvalue()


def grpc_channel_options(
    max_receive_message_length: int = MAX_MESSAGE_SIZE,
    max_send_message_length: int = MAX_MESSAGE_SIZE,
    enable_retries: bool = True,
    initial_backoff: str = "0.1s",
    max_attempts: int = 5,
    backoff_multiplier: float = 2,
    max_backoff: str = "2s",
):
    service_config = {
        "methodConfig": [
            {
                "name": [{}],  # Applies to ALL methods in ALL services
                "retryPolicy": {
                    "maxAttempts": max_attempts,  # Max retries (total attempts = 5)
                    "initialBackoff": initial_backoff,  # First retry after 0.1s
                    "maxBackoff": max_backoff,  # Max wait time between retries
                    "backoffMultiplier": backoff_multiplier,  # Exponential backoff factor
                    "retryableStatusCodes": [
                        "UNAVAILABLE",
                        "DEADLINE_EXCEEDED",
                    ],  # Retries on network failures
                },
            }
        ]
    }

    service_config_json = json.dumps(service_config)

    retries_option = 1 if enable_retries else 0

    return [
        ("grpc.max_receive_message_length", max_receive_message_length),
        ("grpc.max_send_message_length", max_send_message_length),
        ("grpc.enable_retries", retries_option),
        ("grpc.service_config", service_config_json),
    ]
