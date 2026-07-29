#!/usr/bin/env python

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

import io
import pathlib
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Event, Queue
from pickle import UnpicklingError

import pytest
import torch

from lerobot.utils.constants import ACTION
from lerobot.utils.transition import Transition
from tests.utils import require_cuda, skip_if_package_missing


@skip_if_package_missing("grpcio", "grpc")
def test_bytes_buffer_size_empty_buffer():
    from lerobot.transport.utils import bytes_buffer_size

    """Test with an empty buffer."""
    buffer = io.BytesIO()
    assert bytes_buffer_size(buffer) == 0
    # Ensure position is reset to beginning
    assert buffer.tell() == 0


@skip_if_package_missing("grpcio", "grpc")
def test_bytes_buffer_size_small_buffer():
    from lerobot.transport.utils import bytes_buffer_size

    """Test with a small buffer."""
    buffer = io.BytesIO(b"Hello, World!")
    assert bytes_buffer_size(buffer) == 13
    assert buffer.tell() == 0


@skip_if_package_missing("grpcio", "grpc")
def test_bytes_buffer_size_large_buffer():
    from lerobot.transport.utils import CHUNK_SIZE, bytes_buffer_size

    """Test with a large buffer."""
    data = b"x" * (CHUNK_SIZE * 2 + 1000)
    buffer = io.BytesIO(data)
    assert bytes_buffer_size(buffer) == len(data)
    assert buffer.tell() == 0


@skip_if_package_missing("grpcio", "grpc")
def test_send_bytes_in_chunks_empty_data():
    from lerobot.transport.utils import send_bytes_in_chunks, services_pb2

    """Test sending empty data."""
    message_class = services_pb2.InteractionMessage
    chunks = list(send_bytes_in_chunks(b"", message_class))
    assert len(chunks) == 0


@skip_if_package_missing("grpcio", "grpc")
def test_single_chunk_small_data():
    from lerobot.transport.utils import send_bytes_in_chunks, services_pb2

    """Test data that fits in a single chunk."""
    data = b"Some data"
    message_class = services_pb2.InteractionMessage
    chunks = list(send_bytes_in_chunks(data, message_class))

    assert len(chunks) == 1
    assert chunks[0].data == b"Some data"
    assert chunks[0].transfer_state == services_pb2.TransferState.TRANSFER_END


@skip_if_package_missing("grpcio", "grpc")
def test_not_silent_mode():
    from lerobot.transport.utils import send_bytes_in_chunks, services_pb2

    """Test not silent mode."""
    data = b"Some data"
    message_class = services_pb2.InteractionMessage
    chunks = list(send_bytes_in_chunks(data, message_class, silent=False))
    assert len(chunks) == 1
    assert chunks[0].data == b"Some data"


@skip_if_package_missing("grpcio", "grpc")
def test_send_bytes_in_chunks_large_data():
    from lerobot.transport.utils import CHUNK_SIZE, send_bytes_in_chunks, services_pb2

    """Test sending large data."""
    data = b"x" * (CHUNK_SIZE * 2 + 1000)
    message_class = services_pb2.InteractionMessage
    chunks = list(send_bytes_in_chunks(data, message_class))
    assert len(chunks) == 3
    assert chunks[0].data == b"x" * CHUNK_SIZE
    assert chunks[0].transfer_state == services_pb2.TransferState.TRANSFER_BEGIN
    assert chunks[1].data == b"x" * CHUNK_SIZE
    assert chunks[1].transfer_state == services_pb2.TransferState.TRANSFER_MIDDLE
    assert chunks[2].data == b"x" * 1000
    assert chunks[2].transfer_state == services_pb2.TransferState.TRANSFER_END


@skip_if_package_missing("grpcio", "grpc")
def test_send_bytes_in_chunks_large_data_with_exact_chunk_size():
    from lerobot.transport.utils import CHUNK_SIZE, send_bytes_in_chunks, services_pb2

    """Test sending large data with exact chunk size."""
    data = b"x" * CHUNK_SIZE
    message_class = services_pb2.InteractionMessage
    chunks = list(send_bytes_in_chunks(data, message_class))
    assert len(chunks) == 1
    assert chunks[0].data == data
    assert chunks[0].transfer_state == services_pb2.TransferState.TRANSFER_END


@skip_if_package_missing("grpcio", "grpc")
def test_receive_bytes_in_chunks_empty_data():
    from lerobot.transport.utils import receive_bytes_in_chunks

    """Test receiving empty data."""
    queue = Queue()
    shutdown_event = Event()

    # Empty iterator
    receive_bytes_in_chunks(iter([]), queue, shutdown_event)

    assert queue.empty()


@skip_if_package_missing("grpcio", "grpc")
def test_receive_bytes_in_chunks_single_chunk():
    from lerobot.transport.utils import receive_bytes_in_chunks, services_pb2

    """Test receiving a single chunk message."""
    queue = Queue()
    shutdown_event = Event()

    data = b"Single chunk data"
    chunks = [
        services_pb2.InteractionMessage(data=data, transfer_state=services_pb2.TransferState.TRANSFER_END)
    ]

    receive_bytes_in_chunks(iter(chunks), queue, shutdown_event)

    assert queue.get(timeout=0.01) == data
    assert queue.empty()


@skip_if_package_missing("grpcio", "grpc")
def test_receive_bytes_in_chunks_single_not_end_chunk():
    from lerobot.transport.utils import receive_bytes_in_chunks, services_pb2

    """Test receiving a single chunk message."""
    queue = Queue()
    shutdown_event = Event()

    data = b"Single chunk data"
    chunks = [
        services_pb2.InteractionMessage(data=data, transfer_state=services_pb2.TransferState.TRANSFER_MIDDLE)
    ]

    receive_bytes_in_chunks(iter(chunks), queue, shutdown_event)

    assert queue.empty()


@skip_if_package_missing("grpcio", "grpc")
def test_receive_bytes_in_chunks_multiple_chunks():
    from lerobot.transport.utils import receive_bytes_in_chunks, services_pb2

    """Test receiving a multi-chunk message."""
    queue = Queue()
    shutdown_event = Event()

    chunks = [
        services_pb2.InteractionMessage(
            data=b"First ", transfer_state=services_pb2.TransferState.TRANSFER_BEGIN
        ),
        services_pb2.InteractionMessage(
            data=b"Middle ", transfer_state=services_pb2.TransferState.TRANSFER_MIDDLE
        ),
        services_pb2.InteractionMessage(data=b"Last", transfer_state=services_pb2.TransferState.TRANSFER_END),
    ]

    receive_bytes_in_chunks(iter(chunks), queue, shutdown_event)

    assert queue.get(timeout=0.01) == b"First Middle Last"
    assert queue.empty()


@skip_if_package_missing("grpcio", "grpc")
def test_receive_bytes_in_chunks_multiple_messages():
    from lerobot.transport.utils import receive_bytes_in_chunks, services_pb2

    """Test receiving multiple complete messages in sequence."""
    queue = Queue()
    shutdown_event = Event()

    chunks = [
        # First message - single chunk
        services_pb2.InteractionMessage(
            data=b"Message1", transfer_state=services_pb2.TransferState.TRANSFER_END
        ),
        # Second message - multi chunk
        services_pb2.InteractionMessage(
            data=b"Start2 ", transfer_state=services_pb2.TransferState.TRANSFER_BEGIN
        ),
        services_pb2.InteractionMessage(
            data=b"Middle2 ", transfer_state=services_pb2.TransferState.TRANSFER_MIDDLE
        ),
        services_pb2.InteractionMessage(data=b"End2", transfer_state=services_pb2.TransferState.TRANSFER_END),
        # Third message - single chunk
        services_pb2.InteractionMessage(
            data=b"Message3", transfer_state=services_pb2.TransferState.TRANSFER_END
        ),
    ]

    receive_bytes_in_chunks(iter(chunks), queue, shutdown_event)

    # Should have three messages in queue
    assert queue.get(timeout=0.01) == b"Message1"
    assert queue.get(timeout=0.01) == b"Start2 Middle2 End2"
    assert queue.get(timeout=0.01) == b"Message3"
    assert queue.empty()


@skip_if_package_missing("grpcio", "grpc")
def test_receive_bytes_in_chunks_shutdown_during_receive():
    from lerobot.transport.utils import receive_bytes_in_chunks, services_pb2

    """Test that shutdown event stops receiving mid-stream."""
    queue = Queue()
    shutdown_event = Event()
    shutdown_event.set()

    chunks = [
        services_pb2.InteractionMessage(
            data=b"First ", transfer_state=services_pb2.TransferState.TRANSFER_BEGIN
        ),
        services_pb2.InteractionMessage(
            data=b"Middle ", transfer_state=services_pb2.TransferState.TRANSFER_MIDDLE
        ),
        services_pb2.InteractionMessage(data=b"Last", transfer_state=services_pb2.TransferState.TRANSFER_END),
    ]

    receive_bytes_in_chunks(iter(chunks), queue, shutdown_event)

    assert queue.empty()


@skip_if_package_missing("grpcio", "grpc")
def test_receive_bytes_in_chunks_only_begin_chunk():
    from lerobot.transport.utils import receive_bytes_in_chunks, services_pb2

    """Test receiving only a BEGIN chunk without END."""
    queue = Queue()
    shutdown_event = Event()

    chunks = [
        services_pb2.InteractionMessage(
            data=b"Start", transfer_state=services_pb2.TransferState.TRANSFER_BEGIN
        ),
        # No END chunk
    ]

    receive_bytes_in_chunks(iter(chunks), queue, shutdown_event)

    assert queue.empty()


@skip_if_package_missing("grpcio", "grpc")
def test_receive_bytes_in_chunks_missing_begin():
    from lerobot.transport.utils import receive_bytes_in_chunks, services_pb2

    """Test receiving chunks starting with MIDDLE instead of BEGIN."""
    queue = Queue()
    shutdown_event = Event()

    chunks = [
        # Missing BEGIN
        services_pb2.InteractionMessage(
            data=b"Middle", transfer_state=services_pb2.TransferState.TRANSFER_MIDDLE
        ),
        services_pb2.InteractionMessage(data=b"End", transfer_state=services_pb2.TransferState.TRANSFER_END),
    ]

    receive_bytes_in_chunks(iter(chunks), queue, shutdown_event)

    # The implementation continues from where it is, so we should get partial data
    assert queue.get(timeout=0.01) == b"MiddleEnd"
    assert queue.empty()


# Tests for state_to_bytes and bytes_to_state_dict
@skip_if_package_missing("grpcio", "grpc")
def test_state_to_bytes_empty_dict():
    from lerobot.transport.utils import bytes_to_state_dict, state_to_bytes

    """Test converting empty state dict to bytes."""
    state_dict = {}
    data = state_to_bytes(state_dict)
    reconstructed = bytes_to_state_dict(data)
    assert reconstructed == state_dict


@skip_if_package_missing("grpcio", "grpc")
def test_bytes_to_state_dict_empty_data():
    from lerobot.transport.utils import bytes_to_state_dict

    """Test converting empty data to state dict."""
    with pytest.raises(EOFError):
        bytes_to_state_dict(b"")


@skip_if_package_missing("grpcio", "grpc")
def test_state_to_bytes_simple_dict():
    from lerobot.transport.utils import bytes_to_state_dict, state_to_bytes

    """Test converting simple state dict to bytes."""
    state_dict = {
        "layer1.weight": torch.randn(10, 5),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(1, 10),
        "layer2.bias": torch.randn(1),
    }

    data = state_to_bytes(state_dict)
    assert isinstance(data, bytes)
    assert len(data) > 0

    reconstructed = bytes_to_state_dict(data)

    assert len(reconstructed) == len(state_dict)
    for key in state_dict:
        assert key in reconstructed
        assert torch.allclose(state_dict[key], reconstructed[key])


@skip_if_package_missing("grpcio", "grpc")
def test_state_to_bytes_various_dtypes():
    from lerobot.transport.utils import bytes_to_state_dict, state_to_bytes

    """Test converting state dict with various tensor dtypes."""
    state_dict = {
        "float32": torch.randn(5, 5),
        "float64": torch.randn(3, 3).double(),
        "int32": torch.randint(0, 100, (4, 4), dtype=torch.int32),
        "int64": torch.randint(0, 100, (2, 2), dtype=torch.int64),
        "bool": torch.tensor([True, False, True]),
        "uint8": torch.randint(0, 255, (3, 3), dtype=torch.uint8),
    }

    data = state_to_bytes(state_dict)
    reconstructed = bytes_to_state_dict(data)

    for key in state_dict:
        assert reconstructed[key].dtype == state_dict[key].dtype
        if state_dict[key].dtype == torch.bool:
            assert torch.equal(state_dict[key], reconstructed[key])
        else:
            assert torch.allclose(state_dict[key], reconstructed[key])


@skip_if_package_missing("grpcio", "grpc")
def test_bytes_to_state_dict_invalid_data():
    from lerobot.transport.utils import bytes_to_state_dict

    """Test bytes_to_state_dict with invalid data."""
    with pytest.raises(UnpicklingError):
        bytes_to_state_dict(b"This is not a valid torch save file")


@require_cuda
@skip_if_package_missing("grpcio", "grpc")
def test_state_to_bytes_various_dtypes_cuda():
    from lerobot.transport.utils import bytes_to_state_dict, state_to_bytes

    """Test converting state dict with various tensor dtypes."""
    state_dict = {
        "float32": torch.randn(5, 5).cuda(),
        "float64": torch.randn(3, 3).double().cuda(),
        "int32": torch.randint(0, 100, (4, 4), dtype=torch.int32).cuda(),
        "int64": torch.randint(0, 100, (2, 2), dtype=torch.int64).cuda(),
        "bool": torch.tensor([True, False, True]),
        "uint8": torch.randint(0, 255, (3, 3), dtype=torch.uint8),
    }

    data = state_to_bytes(state_dict)
    reconstructed = bytes_to_state_dict(data)

    for key in state_dict:
        assert reconstructed[key].dtype == state_dict[key].dtype
        if state_dict[key].dtype == torch.bool:
            assert torch.equal(state_dict[key], reconstructed[key])
        else:
            assert torch.allclose(state_dict[key], reconstructed[key])


@skip_if_package_missing("grpcio", "grpc")
def test_python_object_to_bytes_none():
    from lerobot.transport.utils import bytes_to_python_object, python_object_to_bytes

    """Test converting None to bytes."""
    obj = None
    data = python_object_to_bytes(obj)
    reconstructed = bytes_to_python_object(data)
    assert reconstructed is None


@pytest.mark.parametrize(
    "obj",
    [
        42,
        -123,
        3.14159,
        -2.71828,
        "Hello, World!",
        "Unicode: 你好世界 🌍",
        True,
        False,
        b"byte string",
        [],
        [1, 2, 3],
        [1, "two", 3.0, True, None],
        {},
        {"key": "value", "number": 123, "nested": {"a": 1}},
        (),
        (1, 2, 3),
    ],
)
@skip_if_package_missing("grpcio", "grpc")
def test_python_object_to_bytes_simple_types(obj):
    from lerobot.transport.utils import bytes_to_python_object, python_object_to_bytes

    """Test converting simple Python types."""
    data = python_object_to_bytes(obj)
    reconstructed = bytes_to_python_object(data)
    assert reconstructed == obj
    assert type(reconstructed) is type(obj)


@skip_if_package_missing("grpcio", "grpc")
def test_python_object_to_bytes_with_tensors():
    from lerobot.transport.utils import bytes_to_python_object, python_object_to_bytes

    """Test converting objects containing PyTorch tensors."""
    obj = {
        "tensor": torch.randn(5, 5),
        "list_with_tensor": [1, 2, torch.randn(3, 3), "string"],
        "nested": {
            "tensor1": torch.randn(2, 2),
            "tensor2": torch.tensor([1, 2, 3]),
        },
    }

    data = python_object_to_bytes(obj)
    reconstructed = bytes_to_python_object(data)

    assert torch.allclose(obj["tensor"], reconstructed["tensor"])
    assert reconstructed["list_with_tensor"][0] == 1
    assert reconstructed["list_with_tensor"][3] == "string"
    assert torch.allclose(obj["list_with_tensor"][2], reconstructed["list_with_tensor"][2])
    assert torch.allclose(obj["nested"]["tensor1"], reconstructed["nested"]["tensor1"])
    assert torch.equal(obj["nested"]["tensor2"], reconstructed["nested"]["tensor2"])


@skip_if_package_missing("grpcio", "grpc")
def test_transitions_to_bytes_empty_list():
    from lerobot.transport.utils import bytes_to_transitions, transitions_to_bytes

    """Test converting empty transitions list."""
    transitions = []
    data = transitions_to_bytes(transitions)
    reconstructed = bytes_to_transitions(data)
    assert reconstructed == transitions
    assert isinstance(reconstructed, list)


@skip_if_package_missing("grpcio", "grpc")
def test_transitions_to_bytes_single_transition():
    from lerobot.transport.utils import bytes_to_transitions, transitions_to_bytes

    """Test converting a single transition."""
    transition = Transition(
        state={"image": torch.randn(3, 64, 64), "state": torch.randn(10)},
        action=torch.randn(5),
        reward=torch.tensor(1.5),
        done=torch.tensor(False),
        next_state={"image": torch.randn(3, 64, 64), "state": torch.randn(10)},
    )

    transitions = [transition]
    data = transitions_to_bytes(transitions)
    reconstructed = bytes_to_transitions(data)

    assert len(reconstructed) == 1

    assert_transitions_equal(transitions[0], reconstructed[0])


@skip_if_package_missing("grpcio", "grpc")
def assert_transitions_equal(t1: Transition, t2: Transition):
    """Helper to assert two transitions are equal."""
    assert_observation_equal(t1["state"], t2["state"])
    assert torch.allclose(t1[ACTION], t2[ACTION])
    assert torch.allclose(t1["reward"], t2["reward"])
    assert torch.equal(t1["done"], t2["done"])
    assert_observation_equal(t1["next_state"], t2["next_state"])


@skip_if_package_missing("grpcio", "grpc")
def assert_observation_equal(o1: dict, o2: dict):
    """Helper to assert two observations are equal."""
    assert set(o1.keys()) == set(o2.keys())
    for key in o1:
        assert torch.allclose(o1[key], o2[key])


@skip_if_package_missing("grpcio", "grpc")
def test_transitions_to_bytes_multiple_transitions():
    from lerobot.transport.utils import bytes_to_transitions, transitions_to_bytes

    """Test converting multiple transitions."""
    transitions = []
    for i in range(5):
        transition = Transition(
            state={"data": torch.randn(10)},
            action=torch.randn(3),
            reward=torch.tensor(float(i)),
            done=torch.tensor(i == 4),
            next_state={"data": torch.randn(10)},
        )
        transitions.append(transition)

    data = transitions_to_bytes(transitions)
    reconstructed = bytes_to_transitions(data)

    assert len(reconstructed) == len(transitions)
    for original, reconstructed_item in zip(transitions, reconstructed, strict=False):
        assert_transitions_equal(original, reconstructed_item)


@skip_if_package_missing("grpcio", "grpc")
def test_receive_bytes_in_chunks_unknown_state():
    from lerobot.transport.utils import receive_bytes_in_chunks

    """Test receive_bytes_in_chunks with an unknown transfer state."""

    # Mock the gRPC message object, which has `transfer_state` and `data` attributes.
    class MockMessage:
        def __init__(self, transfer_state, data):
            self.transfer_state = transfer_state
            self.data = data

    # 10 is not a valid TransferState enum value
    bad_iterator = [MockMessage(transfer_state=10, data=b"bad_data")]
    output_queue = Queue()
    shutdown_event = Event()

    with pytest.raises(ValueError, match="Received unknown transfer state"):
        receive_bytes_in_chunks(bad_iterator, output_queue, shutdown_event)


# --- Security regression: the gRPC transport is unauthenticated plaintext, so
# --- any payload reaching `bytes_to_python_object` is attacker-controlled.


@skip_if_package_missing("grpcio", "grpc")
def test_bytes_to_python_object_rejects_code_execution_payload(tmp_path):
    """A `__reduce__` payload must be refused, not executed.

    This is the regression test for the unauthenticated-RPC RCE: a bare
    `pickle.load` here runs whatever the sender names in `__reduce__`.
    """
    import pickle

    from lerobot.transport.utils import bytes_to_python_object

    marker = tmp_path / "pwned"

    class Exploit:
        def __reduce__(self):
            return (pathlib.Path.write_text, (marker, "pwned"))

    payload = pickle.dumps(Exploit())

    with pytest.raises(UnpicklingError):
        bytes_to_python_object(payload)

    assert not marker.exists(), "deserialization executed attacker-controlled code"


@skip_if_package_missing("grpcio", "grpc")
def test_bytes_to_python_object_rejects_arbitrary_global(tmp_path):
    """Naming a non-allowlisted callable must fail closed."""
    import pickle

    from lerobot.transport.utils import bytes_to_python_object

    payload = pickle.dumps({"callable": pathlib.Path.write_text})

    with pytest.raises(UnpicklingError):
        bytes_to_python_object(payload)


@skip_if_package_missing("grpcio", "grpc")
def test_python_object_to_bytes_roundtrips_numpy():
    """Camera frames cross the wire as numpy arrays, so they must survive."""
    import numpy as np

    from lerobot.transport.utils import bytes_to_python_object, python_object_to_bytes

    obj = {
        "observation.images.front": np.zeros((3, 8, 8), dtype=np.uint8),
        "observation.state": np.arange(6, dtype=np.float32),
    }
    reconstructed = bytes_to_python_object(python_object_to_bytes(obj))

    assert np.array_equal(reconstructed["observation.images.front"], obj["observation.images.front"])
    assert reconstructed["observation.images.front"].dtype == np.uint8
    assert np.array_equal(reconstructed["observation.state"], obj["observation.state"])


@skip_if_package_missing("grpcio", "grpc")
def test_python_object_to_bytes_roundtrips_wire_dataclasses():
    """The async-inference payload types must round-trip through the codec."""
    from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction, TimedObservation
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.transport.utils import bytes_to_python_object, python_object_to_bytes

    obs = TimedObservation(
        timestamp=1.5, timestep=3, observation={"observation.state": torch.randn(6)}, must_go=True
    )
    got = bytes_to_python_object(python_object_to_bytes(obs))
    assert isinstance(got, TimedObservation)
    assert got.timestep == 3 and got.must_go is True
    assert torch.allclose(got.observation["observation.state"], obs.observation["observation.state"])

    actions = [TimedAction(timestamp=float(i), timestep=i, action=torch.randn(6)) for i in range(3)]
    got = bytes_to_python_object(python_object_to_bytes(actions))
    assert len(got) == 3 and all(isinstance(a, TimedAction) for a in got)
    assert torch.allclose(got[1].action, actions[1].action)

    cfg = RemotePolicyConfig(
        policy_type="act",
        pretrained_name_or_path="lerobot/act",
        lerobot_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))},
        actions_per_chunk=50,
    )
    got = bytes_to_python_object(python_object_to_bytes(cfg))
    assert got.lerobot_features["observation.state"].type is FeatureType.STATE
    assert got.actions_per_chunk == 50


@skip_if_package_missing("grpcio", "grpc")
def test_bytes_to_python_object_is_thread_safe():
    """Concurrent deserialization must not fail spuriously.

    `torch.serialization.safe_globals` is a context manager over a module-global
    set with no reference counting, so using it here would let the first thread
    to exit strip the allowlist out from under threads still loading. The policy
    server runs gRPC on four threads and deserializes observations on the hot
    path, so that race drops observations on a robot control loop — and an
    unauthenticated peer opening a second connection can induce it deliberately.
    """
    from lerobot.async_inference.helpers import TimedObservation
    from lerobot.transport.utils import bytes_to_python_object, python_object_to_bytes

    payload = python_object_to_bytes(
        TimedObservation(timestamp=1.0, timestep=1, observation={"observation.state": torch.randn(256)})
    )

    workers = 4  # matches `PolicyServer`'s ThreadPoolExecutor
    failures: list[str] = []

    def load_repeatedly() -> None:
        for _ in range(400):
            try:
                bytes_to_python_object(payload)
            except Exception as e:  # noqa: BLE001 - the failure mode is what we assert on
                failures.append(f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda _: load_repeatedly(), range(workers)))

    assert not failures, f"{len(failures)} concurrent deserializations failed, e.g. {failures[0]}"


@skip_if_package_missing("grpcio", "grpc")
def test_receive_bytes_in_chunks_rejects_oversized_message():
    """An unauthenticated peer must not be able to exhaust host memory.

    gRPC's `MAX_MESSAGE_SIZE` caps each individual frame, but the chunking
    protocol concatenates an unbounded number of them, so only the reassembly
    cap stops a peer that never sends TRANSFER_END.
    """
    from lerobot.transport.utils import receive_bytes_in_chunks, services_pb2

    chunk = b"A" * 1024
    chunks = [
        services_pb2.InteractionMessage(data=chunk, transfer_state=services_pb2.TransferState.TRANSFER_BEGIN)
    ]
    chunks += [
        services_pb2.InteractionMessage(data=chunk, transfer_state=services_pb2.TransferState.TRANSFER_MIDDLE)
        for _ in range(20)
    ]

    with pytest.raises(ValueError, match="exceeds"):
        receive_bytes_in_chunks(iter(chunks), Queue(), Event(), max_message_bytes=4 * 1024)


@skip_if_package_missing("grpcio", "grpc")
def test_receive_bytes_in_chunks_allows_message_at_the_cap():
    """The cap must not reject legitimate payloads just under it."""
    from lerobot.transport.utils import receive_bytes_in_chunks, services_pb2

    chunk = b"B" * 1024
    chunks = [
        services_pb2.InteractionMessage(data=chunk, transfer_state=services_pb2.TransferState.TRANSFER_BEGIN),
        services_pb2.InteractionMessage(
            data=chunk, transfer_state=services_pb2.TransferState.TRANSFER_MIDDLE
        ),
        services_pb2.InteractionMessage(data=chunk, transfer_state=services_pb2.TransferState.TRANSFER_END),
    ]
    queue = Queue()
    receive_bytes_in_chunks(iter(chunks), queue, Event(), max_message_bytes=8 * 1024)
    assert queue.get() == chunk * 3
