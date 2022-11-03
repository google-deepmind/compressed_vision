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

"""Normalize functions constructors."""

from typing import Any, Dict, Optional

import chex
import haiku as hk
from jax import numpy as jnp


class _BatchNorm(hk.BatchNorm):
  """A `hk.BatchNorm` with adapted default arguments."""

  def __init__(self,
               create_scale: bool = True,
               create_offset: bool = True,
               decay_rate: float = 0.9,
               eps: float = 1e-5,
               test_local_stats: bool = False,
               **kwargs):
    # Check args.
    if kwargs.get('cross_replica_axis', None) is not None:
      raise ValueError(
          'Attempting to use \'batch_norm\' normalizer, but specifying '
          '`cross_replica_axis`. This is not supported.')

    self._test_local_stats = test_local_stats
    super().__init__(create_scale=create_scale,
                     create_offset=create_offset,
                     decay_rate=decay_rate,
                     eps=eps,
                     **kwargs)

  def __call__(self,
               x: chex.Array,
               is_training: bool = True) -> jnp.ndarray:
    return super().__call__(x, is_training,
                            test_local_stats=self._test_local_stats)


_NORMALIZER_NAME_TO_CLASS = {
    'batch_norm': _BatchNorm,
}


def get_normalize_fn(
    normalizer_name: str = 'batch_norm',
    normalizer_kwargs: Optional[Dict[str, Any]] = None):
  """Handles NormalizeFn creation.

  These functions are expected to be used as part of Haiku model. On each
  application of the returned normalization_fn, a new Haiku layer will be added
  to the model.

  Args:
    normalizer_name: The name of the normalizer to be constructed.
    normalizer_kwargs: The kwargs passed to the normalizer constructor.

  Returns:
    A `NormalizeFn` that when applied will create a new layer.

  Raises:
    ValueError: If `normalizer_name` is unknown.
  """
  # Check args.
  if normalizer_name not in _NORMALIZER_NAME_TO_CLASS:
    raise ValueError(f'Unrecognized `normalizer_name` {normalizer_name}.')

  normalizer_class = _NORMALIZER_NAME_TO_CLASS[normalizer_name]
  normalizer_kwargs = normalizer_kwargs or dict()

  return lambda *a, **k: normalizer_class(**normalizer_kwargs)(*a, **k)  # pylint: disable=unnecessary-lambda
