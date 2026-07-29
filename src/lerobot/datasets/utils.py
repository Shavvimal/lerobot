#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import contextlib
import dataclasses
import importlib.resources
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

import datasets
import numpy as np
import packaging.version
import torch
from huggingface_hub import DatasetCard, DatasetCardData, HfApi
from huggingface_hub.errors import RevisionNotFoundError

from lerobot.utils.utils import flatten_dict, unflatten_dict

V30_MESSAGE = """
The dataset you requested ({repo_id}) is in {version} format.

We introduced a new format since v3.0 which is not backward compatible with v2.1.
Please, update your dataset to the new format using this command:
```
python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id={repo_id}
```

If you already have a converted version uploaded to the hub, then this error might be because of
an older version in your local cache. Consider deleting the cached version and retrying.

If you encounter a problem, contact LeRobot maintainers on [Discord](https://discord.com/invite/s3KuuzsPFb)
or open an [issue on GitHub](https://github.com/huggingface/lerobot/issues/new/choose).
"""

FUTURE_MESSAGE = """
The dataset you requested ({repo_id}) is only available in {version} format.
As we cannot ensure forward compatibility with it, please update your current version of lerobot.
"""


class CompatibilityError(Exception): ...


class BackwardCompatibilityError(CompatibilityError):
    def __init__(self, repo_id: str, version: packaging.version.Version):
        if version.major == 2 and version.minor == 1:
            message = V30_MESSAGE.format(repo_id=repo_id, version=version)
        else:
            raise NotImplementedError(
                "Contact the maintainer on [Discord](https://discord.com/invite/s3KuuzsPFb)."
            )
        super().__init__(message)


class ForwardCompatibilityError(CompatibilityError):
    def __init__(self, repo_id: str, version: packaging.version.Version):
        message = FUTURE_MESSAGE.format(repo_id=repo_id, version=version)
        super().__init__(message)


logger = logging.getLogger(__name__)


DEFAULT_CHUNK_SIZE = 1000  # Max number of files per chunk
DEFAULT_DATA_FILE_SIZE_IN_MB = 100  # Max size per file
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200  # Max size per file

INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"

EPISODES_DIR = "meta/episodes"
DATA_DIR = "data"
VIDEO_DIR = "videos"

CHUNK_FILE_PATTERN = "chunk-{chunk_index:03d}/file-{file_index:03d}"
IMAGE_FILE_PATTERN = "frame-{frame_index:06d}.png"
DEPTH_FILE_PATTERN = "frame-{frame_index:06d}.tiff"
DEFAULT_TASKS_PATH = "meta/tasks.parquet"
DEFAULT_EPISODES_PATH = EPISODES_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"
DEFAULT_DATA_PATH = DATA_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"
DEFAULT_VIDEO_PATH = VIDEO_DIR + "/{video_key}/" + CHUNK_FILE_PATTERN + ".mp4"
DEFAULT_IMAGE_PATH = "images/{image_key}/episode-{episode_index:06d}/" + IMAGE_FILE_PATTERN
DEFAULT_DEPTH_PATH = "images/{image_key}/episode-{episode_index:06d}/" + DEPTH_FILE_PATTERN

LEGACY_EPISODES_PATH = "meta/episodes.jsonl"
LEGACY_EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
LEGACY_TASKS_PATH = "meta/tasks.jsonl"


def _reject_path_escape(field_name: str, value: str, reported_as: str | None = None) -> None:
    """Raise if ``value`` would resolve outside the dataset root once joined onto it.

    Both path flavours are checked. ``PurePosixPath`` treats a backslash as an
    ordinary character, so it reads ``..\\..\\x`` as a single filename, while
    ``PureWindowsPath`` splits it into traversal segments — and the reverse holds
    for a leading ``/``, which only ``PurePosixPath`` calls absolute. Checking one
    flavour therefore leaves the other's traversals through.

    Args:
        field_name: Name used in the error message.
        value: The path fragment to check.
        reported_as: Value to quote in the error instead of ``value`` — used to
            report the original template rather than a formatted probe.

    Raises:
        ValueError: if ``value`` is absolute or contains a ``..`` segment.
    """
    shown = value if reported_as is None else reported_as

    for pure in (PurePosixPath(value), PureWindowsPath(value)):
        if pure.is_absolute():
            raise ValueError(
                f"{field_name} must be a relative path, got {shown!r}. "
                f"Absolute paths escape the dataset root."
            )
        if ".." in pure.parts:
            raise ValueError(
                f"{field_name} must not contain '..' segments, got {shown!r}. "
                f"Such paths escape the dataset root."
            )


def _validate_path_template(field_name: str, template: str) -> None:
    """Reject path templates that would resolve outside the dataset root.

    ``data_path`` / ``video_path`` are read verbatim from ``meta/info.json``,
    which ships inside a dataset repository and is downloaded from the Hub.
    Callers join the formatted result onto the dataset root, so an absolute
    template silently discards the root (``Path("/root") / "/tmp/x"`` is
    ``/tmp/x``) and a ``..`` template walks out of it. Both the read paths and
    the writer — which does ``mkdir(parents=True)`` and then writes a parquet
    file at the joined location — would follow it.

    Note the ``{video_key}`` / ``{image_key}`` placeholders are substituted from
    the ``features`` dict of the *same* untrusted file, so a probe value proves
    nothing about them; those keys are validated separately in
    :meth:`DatasetInfo.__post_init__`.

    Raises:
        ValueError: if the template is absolute or contains a ``..`` segment.
    """
    if not isinstance(template, str) or not template:
        raise ValueError(f"{field_name} must be a non-empty string, got {template!r}")

    # Check the raw template and a formatted probe: the placeholders themselves
    # are substituted with caller-controlled values, but a traversal hidden in
    # the literal portion has to appear in one of the two.
    candidates = [template]
    # An unknown placeholder just means the probe is skipped — the literal
    # segments of the raw template are still checked below.
    with contextlib.suppress(KeyError, IndexError):
        candidates.append(template.format(chunk_index=0, file_index=0, video_key="probe", episode_index=0))

    for candidate in candidates:
        _reject_path_escape(field_name, candidate, reported_as=template)


@dataclass
class DatasetInfo:
    """Typed representation of the ``meta/info.json`` file for a LeRobot dataset.

    Replaces the previously untyped ``dict`` returned by ``load_info()`` and
    created by ``create_empty_dataset_info()``.  Using a dataclass provides
    explicit field definitions, IDE auto-completion, and validation at
    construction time.
    """

    codebase_version: str
    fps: int
    features: dict[str, dict]

    # Episode / frame counters — start at zero for new datasets
    total_episodes: int = 0
    total_frames: int = 0
    total_tasks: int = 0

    # Storage settings
    chunks_size: int = field(default=DEFAULT_CHUNK_SIZE)
    data_files_size_in_mb: int = field(default=DEFAULT_DATA_FILE_SIZE_IN_MB)
    video_files_size_in_mb: int = field(default=DEFAULT_VIDEO_FILE_SIZE_IN_MB)

    # File path templates
    data_path: str = field(default=DEFAULT_DATA_PATH)
    video_path: str | None = field(default=DEFAULT_VIDEO_PATH)

    # Optional metadata
    robot_type: str | None = None
    splits: dict[str, str] = field(default_factory=dict)
    # OpenAI-style tool schemas declared by the dataset. ``None`` means the
    # dataset doesn't declare any — readers fall back to ``DEFAULT_TOOLS``.
    tools: list[dict] | None = None

    def __post_init__(self) -> None:
        # Coerce feature shapes from list to tuple — JSON deserialisation
        # returns lists, but the rest of the codebase expects tuples.
        for ft in self.features.values():
            if isinstance(ft.get("shape"), list):
                ft["shape"] = tuple(ft["shape"])

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.chunks_size <= 0:
            raise ValueError(f"chunks_size must be positive, got {self.chunks_size}")
        if self.data_files_size_in_mb <= 0:
            raise ValueError(f"data_files_size_in_mb must be positive, got {self.data_files_size_in_mb}")
        if self.video_files_size_in_mb <= 0:
            raise ValueError(f"video_files_size_in_mb must be positive, got {self.video_files_size_in_mb}")

        _validate_path_template("data_path", self.data_path)
        if self.video_path is not None:
            _validate_path_template("video_path", self.video_path)

        # Feature keys are substituted into `video_path` as `{video_key}` and into
        # `DEFAULT_IMAGE_PATH` as `{image_key}`, then joined onto the dataset root
        # by the readers and by `DatasetWriter`. They come from the same untrusted
        # `meta/info.json` as the templates, so validating the template alone still
        # leaves `videos/{video_key}/...` with a key of `../../../tmp/evil` able to
        # escape. Only traversal is rejected — a key containing a plain `/` merely
        # nests a directory, and rejecting it would break existing datasets.
        for key in self.features:
            _reject_path_escape("feature key", key)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict.

        Converts tuple shapes back to lists so ``json.dump`` can handle them.
        Drops ``tools`` when unset so existing datasets keep a clean
        ``info.json``.
        """
        d = dataclasses.asdict(self)
        for ft in d["features"].values():
            if isinstance(ft.get("shape"), tuple):
                ft["shape"] = list(ft["shape"])
        if d.get("tools") is None:
            d.pop("tools", None)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetInfo":
        """Construct from a raw dict (e.g. loaded directly from JSON).

        Unknown keys are ignored for forward compatibility with datasets that
        carry additional fields (e.g. ``total_videos`` from v2.x). A warning is
        logged when such fields are present.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(k for k in data if k not in known)
        if unknown:
            logger.warning(f"Unknown fields in DatasetInfo: {unknown}. These will be ignored.")
        return cls(**{k: v for k, v in data.items() if k in known})

    # ---------------------------------------------------------------------------
    # Temporary dict-style compatibility layer
    # Allows existing ``info["key"]`` call-sites to keep working without changes.
    # Once all callers have been migrated to attribute access, remove these.
    # ---------------------------------------------------------------------------
    def __getitem__(self, key: str):
        import warnings

        warnings.warn(
            f"Accessing DatasetInfo with dict-style syntax info['{key}'] is deprecated. "
            f"Use attribute access info.{key} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            return getattr(self, key)
        except AttributeError as err:
            raise KeyError(key) from err

    def __setitem__(self, key: str, value) -> None:
        import warnings

        warnings.warn(
            f"Setting DatasetInfo with dict-style syntax info['{key}'] = ... is deprecated. "
            f"Use attribute assignment info.{key} = ... instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not hasattr(self, key):
            raise KeyError(f"DatasetInfo has no field '{key}'")
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Check if a field exists (dict-like interface)."""
        return hasattr(self, key)

    def get(self, key: str, default=None):
        """Get attribute value with default fallback (dict-like interface)."""
        try:
            return getattr(self, key)
        except AttributeError:
            return default


def has_legacy_hub_download_metadata(root: Path) -> bool:
    """Return ``True`` when *root* looks like a legacy Hub ``local_dir`` mirror.

    ``snapshot_download(local_dir=...)`` stores lightweight metadata under
    ``<local_dir>/.cache/huggingface/download/``.  The presence of this
    directory is a reliable indicator that the dataset was downloaded with
    the old non-revision-safe ``local_dir`` mode and should be re-fetched
    through the snapshot cache instead.
    """
    return (root / ".cache" / "huggingface" / "download").exists()


def update_chunk_file_indices(chunk_idx: int, file_idx: int, chunks_size: int) -> tuple[int, int]:
    if file_idx == chunks_size - 1:
        file_idx = 0
        chunk_idx += 1
    else:
        file_idx += 1
    return chunk_idx, file_idx


def serialize_dict(stats: dict[str, torch.Tensor | np.ndarray | dict]) -> dict:
    """Serialize a dictionary containing tensors or numpy arrays to be JSON-compatible.

    Converts torch.Tensor, np.ndarray, and np.generic types to lists or native Python types.

    Args:
        stats (dict): A dictionary that may contain non-serializable numeric types.

    Returns:
        dict: A dictionary with all values converted to JSON-serializable types.

    Raises:
        NotImplementedError: If a value has an unsupported type.
    """
    serialized_dict = {}
    for key, value in flatten_dict(stats).items():
        if isinstance(value, (torch.Tensor | np.ndarray)):
            serialized_dict[key] = value.tolist()
        elif isinstance(value, list) and isinstance(value[0], (int | float | list)):
            serialized_dict[key] = value
        elif isinstance(value, np.generic):
            serialized_dict[key] = value.item()
        elif isinstance(value, (int | float)):
            serialized_dict[key] = value
        else:
            raise NotImplementedError(f"The value '{value}' of type '{type(value)}' is not supported.")
    return unflatten_dict(serialized_dict)


def is_valid_version(version: str) -> bool:
    """Check if a string is a valid PEP 440 version.

    Args:
        version (str): The version string to check.

    Returns:
        bool: True if the version string is valid, False otherwise.
    """
    try:
        packaging.version.parse(version)
        return True
    except packaging.version.InvalidVersion:
        return False


def check_version_compatibility(
    repo_id: str,
    version_to_check: str | packaging.version.Version,
    current_version: str | packaging.version.Version,
    enforce_breaking_major: bool = True,
) -> None:
    """Check for version compatibility between a dataset and the current codebase.

    Args:
        repo_id (str): The repository ID for logging purposes.
        version_to_check (str | packaging.version.Version): The version of the dataset.
        current_version (str | packaging.version.Version): The current version of the codebase.
        enforce_breaking_major (bool): If True, raise an error on major version mismatch.

    Raises:
        BackwardCompatibilityError: If the dataset version is from a newer, incompatible
            major version of the codebase.
    """
    v_check = (
        packaging.version.parse(version_to_check)
        if not isinstance(version_to_check, packaging.version.Version)
        else version_to_check
    )
    v_current = (
        packaging.version.parse(current_version)
        if not isinstance(current_version, packaging.version.Version)
        else current_version
    )
    if v_check.major < v_current.major and enforce_breaking_major:
        raise BackwardCompatibilityError(repo_id, v_check)
    elif v_check.minor < v_current.minor:
        logging.warning(FUTURE_MESSAGE.format(repo_id=repo_id, version=v_check))


def get_repo_versions(repo_id: str, *, token: str | bool | None = None) -> list[packaging.version.Version]:
    """Return available valid versions (branches and tags) on a given Hub repo.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.
        token: Authentication token used for Hub requests. Pass a string token,
            ``True`` to require the locally stored token, ``False`` to disable
            authentication, or ``None`` to use the Hugging Face Hub default.

    Returns:
        list[packaging.version.Version]: A list of valid versions found.
    """
    api = HfApi() if token is None else HfApi(token=token)
    repo_refs = api.list_repo_refs(repo_id, repo_type="dataset")
    repo_refs = [b.name for b in repo_refs.branches + repo_refs.tags]
    repo_versions = []
    for ref in repo_refs:
        with contextlib.suppress(packaging.version.InvalidVersion):
            repo_versions.append(packaging.version.parse(ref))

    return repo_versions


def get_safe_version(
    repo_id: str,
    version: str | packaging.version.Version,
    *,
    token: str | bool | None = None,
) -> str:
    """Return the specified version if available on repo, or the latest compatible one.

    If the exact version is not found, it looks for the latest version with the
    same major version number that is less than or equal to the target minor version.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.
        version (str | packaging.version.Version): The target version.
        token: Authentication token forwarded to the Hub version lookup.

    Returns:
        str: The safe version string (e.g., "v1.2.3") to use as a revision.

    Raises:
        RevisionNotFoundError: If the repo has no version tags.
        BackwardCompatibilityError: If only older major versions are available.
        ForwardCompatibilityError: If only newer major versions are available.
    """
    target_version = (
        packaging.version.parse(version) if not isinstance(version, packaging.version.Version) else version
    )
    hub_versions = get_repo_versions(repo_id) if token is None else get_repo_versions(repo_id, token=token)

    if not hub_versions:
        raise RevisionNotFoundError(
            f"""Your dataset must be tagged with a codebase version.
            Assuming _version_ is the codebase_version value in the info.json, you can run this:
            ```python
            from huggingface_hub import HfApi

            hub_api = HfApi()
            hub_api.create_tag("{repo_id}", tag="_version_", repo_type="dataset")
            ```
            """
        )

    if target_version in hub_versions:
        return f"v{target_version}"

    compatibles = [
        v for v in hub_versions if v.major == target_version.major and v.minor <= target_version.minor
    ]
    if compatibles:
        return_version = max(compatibles)
        if return_version < target_version:
            logging.warning(f"Revision {version} for {repo_id} not found, using version v{return_version}")
        return f"v{return_version}"

    lower_major = [v for v in hub_versions if v.major < target_version.major]
    if lower_major:
        raise BackwardCompatibilityError(repo_id, max(lower_major))

    upper_versions = [v for v in hub_versions if v > target_version]
    assert len(upper_versions) > 0
    raise ForwardCompatibilityError(repo_id, min(upper_versions))


def create_branch(repo_id: str, *, branch: str, repo_type: str | None = None) -> None:
    """Create a branch on an existing Hugging Face repo.

    Deletes the branch if it already exists before creating it.

    Args:
        repo_id (str): The ID of the repository.
        branch (str): The name of the branch to create.
        repo_type (str | None): The type of the repository (e.g., "dataset").
    """
    api = HfApi()

    branches = api.list_repo_refs(repo_id, repo_type=repo_type).branches
    refs = [branch.ref for branch in branches]
    ref = f"refs/heads/{branch}"
    if ref in refs:
        api.delete_branch(repo_id, repo_type=repo_type, branch=branch)

    api.create_branch(repo_id, repo_type=repo_type, branch=branch)


def create_lerobot_dataset_card(
    tags: list | None = None,
    dataset_info: DatasetInfo | None = None,
    **kwargs,
) -> DatasetCard:
    """Create a `DatasetCard` for a LeRobot dataset.

    Keyword arguments are used to replace values in the card template.
    Note: If specified, `license` must be a valid license identifier from
    https://huggingface.co/docs/hub/repositories-licenses.

    Args:
        tags (list | None): A list of tags to add to the dataset card.
        dataset_info (DatasetInfo | None): The dataset's info object, which will
            be displayed on the card.
        **kwargs: Additional keyword arguments to populate the card template.

    Returns:
        DatasetCard: The generated dataset card object.
    """
    card_tags = ["LeRobot"]

    if tags:
        card_tags += tags
    if dataset_info:
        dataset_structure = "[meta/info.json](meta/info.json):\n"
        dataset_structure += f"```json\n{json.dumps(dataset_info.to_dict(), indent=4)}\n```\n"
        kwargs = {**kwargs, "dataset_structure": dataset_structure}
    card_data = DatasetCardData(
        license=kwargs.get("license"),
        tags=card_tags,
        task_categories=["robotics"],
        configs=[
            {
                "config_name": "default",
                "data_files": "data/*/*.parquet",
            }
        ],
    )

    card_template = (importlib.resources.files("lerobot.datasets") / "card_template.md").read_text()

    return DatasetCard.from_template(
        card_data=card_data,
        template_str=card_template,
        **kwargs,
    )


def is_float_in_list(target, float_list, threshold=1e-6):
    return any(abs(target - x) <= threshold for x in float_list)


def find_float_index(target, float_list, threshold=1e-6):
    for i, x in enumerate(float_list):
        if abs(target - x) <= threshold:
            return i
    return -1


def safe_shard(dataset: datasets.IterableDataset, index: int, num_shards: int) -> datasets.Dataset:
    """
    Safe shards the dataset.
    """
    shard_idx = min(dataset.num_shards, index + 1) - 1

    return dataset.shard(num_shards, index=shard_idx)
