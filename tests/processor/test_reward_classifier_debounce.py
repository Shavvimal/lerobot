#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import pytest
import torch
from torch import nn

from lerobot.lerobot_types import TransitionKey
from lerobot.processor import DataProcessorPipeline, RewardClassifierProcessorStep
from lerobot.processor.converters import create_transition
from lerobot.utils.constants import OBS_IMAGE

# A static scene near the success boundary: one noise spike over the threshold, then a real sustained success.
NOISY_THEN_SUCCESS = [0.1, 0.1, 0.9, 0.1, 0.95, 0.95, 0.95]


class SequenceClassifier(nn.Module):
    """Stand-in for `Classifier` that returns a scripted success probability on each call."""

    def __init__(self, probabilities):
        super().__init__()
        self.probabilities = list(probabilities)

    def predict_reward(self, batch, threshold=0.5):
        return (torch.tensor([self.probabilities.pop(0)]) > threshold).float()


def run_episode(step, num_frames):
    """Feeds `num_frames` image observations through the step and returns the (reward, done) pairs."""
    outputs = []
    for _ in range(num_frames):
        transition = create_transition(
            observation={OBS_IMAGE: torch.rand(1, 3, 8, 8)}, reward=0.0, done=False
        )
        result = step(transition)
        outputs.append((result[TransitionKey.REWARD], result[TransitionKey.DONE]))
    return outputs


def first_done_index(outputs):
    return next((i for i, (_, done) in enumerate(outputs) if done), None)


def test_reward_classifier_default_fires_on_first_success_frame():
    step = RewardClassifierProcessorStep(success_threshold=0.7)
    step.reward_classifier = SequenceClassifier(NOISY_THEN_SUCCESS)

    outputs = run_episode(step, len(NOISY_THEN_SUCCESS))

    # Default behaviour is unchanged: the single noisy frame at index 2 ends the episode with the success reward.
    assert first_done_index(outputs) == 2
    assert outputs[2] == (step.success_reward, True)


def test_reward_classifier_requires_consecutive_success_frames():
    step = RewardClassifierProcessorStep(success_threshold=0.7, min_consecutive_success=3)
    step.reward_classifier = SequenceClassifier(NOISY_THEN_SUCCESS)

    outputs = run_episode(step, len(NOISY_THEN_SUCCESS))

    assert [reward for reward, _ in outputs[:6]] == [0.0] * 6
    assert first_done_index(outputs) == 6
    assert outputs[6] == (step.success_reward, True)


def test_reward_classifier_keeps_rewarding_while_streak_holds():
    step = RewardClassifierProcessorStep(
        success_threshold=0.7, min_consecutive_success=2, terminate_on_success=False
    )
    step.reward_classifier = SequenceClassifier([0.9, 0.9, 0.9, 0.1, 0.9])

    outputs = run_episode(step, 5)

    assert [reward for reward, _ in outputs] == [0.0, 1.0, 1.0, 0.0, 0.0]
    assert not any(done for _, done in outputs)


def test_reward_classifier_reset_clears_success_streak():
    step = RewardClassifierProcessorStep(success_threshold=0.7, min_consecutive_success=3)
    step.reward_classifier = SequenceClassifier([0.9, 0.9, 0.9])

    run_episode(step, 2)
    assert step.success_streak == 2
    step.reset()

    # A streak must not carry over from the previous episode.
    assert first_done_index(run_episode(step, 1)) is None


def test_reward_classifier_rejects_non_positive_min_consecutive_success():
    with pytest.raises(ValueError, match="min_consecutive_success"):
        RewardClassifierProcessorStep(min_consecutive_success=0)


def test_reward_classifier_config_round_trip(tmp_path):
    step = RewardClassifierProcessorStep(success_threshold=0.7, min_consecutive_success=3)
    assert step.get_config()["min_consecutive_success"] == 3
    assert "success_streak" not in step.get_config()

    pipeline = DataProcessorPipeline(steps=[step], name="reward_classifier")
    pipeline.save_pretrained(tmp_path)
    loaded = DataProcessorPipeline.from_pretrained(tmp_path, config_filename="reward_classifier.json")

    restored = loaded.steps[0]
    assert isinstance(restored, RewardClassifierProcessorStep)
    assert restored.get_config() == step.get_config()
