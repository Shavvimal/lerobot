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

"""Explicit consent for executing remote code pulled from the Hugging Face Hub.

Several processor steps load a tokenizer, processor or model by repo id and pass
``trust_remote_code=True``, which executes Python shipped in that repo. Until now
the flag was read from the step's own serialized config — that is, from
``preprocessor_config.json`` inside the very repository it is meant to protect
against. A downloaded config could therefore both name the repo to load *and*
declare itself trusted.

Consent has to come from the caller instead. This module holds that decision for
the duration of a load, mirroring the pattern already used for environments in
``lerobot.envs.factory.make_env`` -> ``lerobot.envs.utils._download_hub_file``,
where remote code is refused unless the caller passed ``trust_remote_code=True``.

A :class:`~contextvars.ContextVar` is used rather than a plain global so the
setting does not leak across threads — the policy server loads pipelines on gRPC
worker threads, and one client opting in must not widen another's load.
"""

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar

_TRUST_REMOTE_CODE: ContextVar[bool] = ContextVar("lerobot_processor_trust_remote_code", default=False)


@contextlib.contextmanager
def trust_remote_code_scope(trust_remote_code: bool) -> Iterator[None]:
    """Set the caller's remote-code consent for the duration of the block.

    Args:
        trust_remote_code: Whether steps built inside this block may execute code
            downloaded from the Hub.
    """
    token = _TRUST_REMOTE_CODE.set(trust_remote_code)
    try:
        yield
    finally:
        _TRUST_REMOTE_CODE.reset(token)


def remote_code_allowed() -> bool:
    """Return whether the current caller has consented to executing remote code."""
    return _TRUST_REMOTE_CODE.get()


def require_remote_code_consent(repo_id: str, component: str) -> None:
    """Raise unless the caller has consented to executing remote code.

    Args:
        repo_id: The Hub repository whose code would be executed.
        component: Human-readable name of the thing being loaded, used in the error.

    Raises:
        RuntimeError: if the caller has not opted in.
    """
    if remote_code_allowed():
        return

    raise RuntimeError(
        f"Refusing to load {component} from '{repo_id}': it requires executing Python code "
        f"downloaded from the Hub, and remote code is not trusted by default.\n"
        f"Processor configs can originate from untrusted repositories, and a config is not "
        f"allowed to declare itself trusted, so this must be enabled by the caller.\n"
        f"If you trust '{repo_id}' and have reviewed its code, pass "
        f"`trust_remote_code=True` to `from_pretrained(...)` / `make_pre_post_processors(...)`, "
        f"and pin a `revision` so the code cannot change under you."
    )
