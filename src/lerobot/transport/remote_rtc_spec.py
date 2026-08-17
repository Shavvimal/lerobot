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

"""Wire spec shared by the remote-RTC inference client and server.

The service is registered via gRPC generic handlers with raw-bytes
(pass-through) serializers, so no protobuf message types or generated
stubs are involved.  Every payload is a plain ``dict`` encoded with
:func:`lerobot.transport.utils.python_object_to_bytes` and decoded with
the weights-only :func:`lerobot.transport.utils.bytes_to_python_object`
(dicts, strings, numbers, ``None``, ``torch.Tensor`` and ``np.ndarray``
all pass the safe unpickler natively — no allowlist changes required).

Both sides import this module so method paths cannot drift.
"""

from __future__ import annotations

# Bump when a payload field changes meaning or shape. The server rejects
# mismatched clients at Setup so a stale checkout fails fast and loudly.
WIRE_VERSION = 1

SERVICE_NAME = "lerobot.RemoteRTC"

SETUP = "Setup"
PREDICT = "Predict"
RESET = "Reset"
READY = "Ready"

METHOD_NAMES = (SETUP, PREDICT, RESET, READY)


def method_path(name: str) -> str:
    """Full gRPC method path for a RemoteRTC method name."""
    return f"/{SERVICE_NAME}/{name}"
