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

"""A Haiku I3D Unit.

The model is introduced in:

  Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
  Joao Carreira, Andrew Zisserman
  https://arxiv.org/pdf/1705.07750v1.pdf.

"""

from typing import Callable, Optional, Sequence

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class _MaxPool(hk.MaxPool):
  """A `hk.MaxPool` accepting (and discarding) an `is_training` argument."""

  def __call__(self,
               x: chex.Array,
               is_training: bool = True) -> jnp.ndarray:
    del is_training  # Unused.
    return super().__call__(x)


class Unit3D(hk.Module):
  """Basic I3D unit containing Conv3D + Normalization + non-linearity."""

  def __init__(self,
               output_channels: int,
               kernel_shape: Sequence[int] = (1, 1, 1),
               stride: Sequence[int] = (1, 1, 1),
               with_bias: bool = False,
               normalize_fn: Optional[Callable[..., chex.Array]] = None,
               activation_fn: Optional[Callable[[chex.Array],
                                                chex.Array]] = jax.nn.relu,
               padding: str = 'SAME',
               name: str = 'Unit3D'):
    """Initializes the Unit3D module.

    Args:
      output_channels: Number of output channels.
      kernel_shape: The shape of the kernel. A sequence of length 3.
      stride: Stride for the kernel. A sequence of length 3.
      with_bias: Whether to add a bias to the convolution.
      normalize_fn: Function used for normalization.
      activation_fn: Function used as non-linearity.
      padding: Which type of padding to use (default `SAME`).
      name: The name of the module.

    Raises:
      ValueError: If `kernel_shape` or `stride` has the wrong shape.
    """
    super().__init__(name=name)

    # Check args.
    if len(kernel_shape) != 3:
      raise ValueError(
          'Given `kernel_shape` must have length 3 but has length '
          f'{len(kernel_shape)}.')
    if len(stride) != 3:
      raise ValueError(
          f'Given `stride` must have length 3 but has length {len(stride)}.')

    self._output_channels = output_channels
    self._kernel_shape = kernel_shape
    self._stride = stride
    self._with_bias = with_bias
    self._normalize_fn = normalize_fn
    self._activation_fn = activation_fn
    self._padding = padding

  def __call__(self,
               inputs: chex.Array,
               is_training: bool = True) -> jnp.ndarray:
    """Connects the module to inputs.

    Args:
      inputs: A 5-D float array of shape `[B, T, H, W, C]`.
      is_training: Whether to use training mode.

    Returns:
      A 5-D float array of shape `[B, new_t, new_h, new_w, output_channels]`.
    """
    if self._padding == 'VALID':
      t = (self._kernel_shape[0] - 1) // 2
      h = (self._kernel_shape[1] - 1) // 2
      w = (self._kernel_shape[2] - 1) // 2
      # Needs to be numpy so it does not become a traced array.
      pad_dims = np.array([[0, 0], [t, t], [h, h], [w, w], [0, 0]])
      inputs = jax.numpy.pad(inputs, pad_dims, mode='reflect')
    out = hk.Conv3D(
        output_channels=self._output_channels,
        kernel_shape=self._kernel_shape,
        stride=self._stride,
        padding=self._padding,
        with_bias=self._with_bias)(
            inputs)

    if self._normalize_fn is not None:
      out = self._normalize_fn(out, is_training=is_training)

    if self._activation_fn is not None:
      out = self._activation_fn(out)

    return out
