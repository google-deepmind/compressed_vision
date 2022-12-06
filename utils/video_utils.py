# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Video utils."""
from typing import Sequence
import numpy as np
from PIL import Image


def video_reshaper(sample_video: np.ndarray) -> np.ndarray:
  """Reshape video into the correct shape."""
  if sample_video.shape[0] == 1:
    v = np.array(sample_video[0])
  elif sample_video.shape[0] == 4:
    # Make a grid of 2x2.
    v = np.concatenate(
        [np.concatenate([sample_video[0], sample_video[1]], axis=1),
         np.concatenate([sample_video[2], sample_video[3]], axis=1)],
        axis=2)
  else:
    # Concatenate all horizontally.
    v = np.concatenate(sample_video, axis=2)
  return v


def save_video(frame_generator: Sequence[np.ndarray], video_path: str):
  """Save video to the given path."""
  first_frame = Image.fromarray(frame_generator[0])
  first_frame.save(
      video_path, save_all=True,
      append_images=[Image.fromarray(frame)
                     for frame in frame_generator[1:]], duration=40, loop=0)
