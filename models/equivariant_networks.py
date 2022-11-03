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

"""Models that apply a transformation on some group of codes."""
import functools
from typing import Optional, Sequence

from compressed_vision.models import transformer
import haiku as hk
import jax
import jax.numpy as jnp


class NeuralTransformation(hk.Module):
  """Uses a transformer to transform a set of latent codes."""

  def __init__(self,
               channels: Sequence[int] = (64, 64, 64),
               hidden_size: int = 128,
               num_hidden_layers: int = 2,
               num_heads: int = 4,
               num_attention_heads: int = 1,
               key_size: int = 128,
               value_size: int = 128,
               intermediate_size: int = 128,
               lambda_v: float = 0.0001,
               output_video_shape: Optional[Sequence[int]] = None,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._lambda = lambda_v
    self._output_video_shape = output_video_shape
    self.channels = channels
    self.neural_transformer = functools.partial(
        transformer.SpatioTemporalTransformerXL,
        num_layers=num_hidden_layers,
        num_heads=num_attention_heads,
        key_size=key_size,
        value_size=value_size,
        ffw_hidden_size=intermediate_size,
        dropout_rate=0.0)

  def __call__(self, video, augmentation):
    batch_size = video.shape[0]
    time = video.shape[1]
    channels = video.shape[-1]
    augmentation = augmentation.reshape((augmentation.shape[0], -1))

    mlp = hk.nets.MLP(tuple(self.channels) + (channels,))
    augmentation = mlp(augmentation).reshape(batch_size, 1, 1, channels)
    video_btsc = video.reshape(batch_size, time, -1, channels)

    augmentation = augmentation.repeat(video_btsc.shape[1], 1)
    augmentation = augmentation.repeat(video_btsc.shape[2], 2)
    all_inputs = jnp.concatenate((augmentation, video_btsc), -1)
    result = self.neural_transformer(
        absolute_position_length=(video_btsc.shape[1], video_btsc.shape[2]),
        d_model=channels*2)(all_inputs)[:, :, :, -channels:]
    result = result.reshape(video.shape)
    if self._output_video_shape is not None:
      result = jax.image.resize(
          result,
          shape=(batch_size, time) + self._output_video_shape,
          method='nearest')
    return result


def get_equivariant_network(network_name):
  if network_name == 'transformer':
    return NeuralTransformation
  else:
    raise ValueError(f'Unknown network: {network_name}.')
