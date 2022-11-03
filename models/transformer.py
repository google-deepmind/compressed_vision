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

"""Transformer-XL implementation."""

import functools
from typing import Any, Callable, Mapping, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def _layer_norm(
    x: jnp.ndarray,
    use_bias: bool = True,
    name: Optional[str] = None,
) -> jnp.ndarray:
  ln = hk.LayerNorm(
      axis=-1, create_scale=True, create_offset=use_bias, name=name)
  return ln(x)


class DenseBlock(hk.Module):
  """Dense block."""

  def __init__(
      self,
      *,
      ffw_hidden_size: int,
      dropout_rate: float,
      init_scale: float,
      final_init_scale_multiplier: float,
      use_final_bias: bool,
      activation: Callable[[jnp.ndarray], jnp.ndarray],
      output_channels: Optional[int] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._ffw_hidden_size = ffw_hidden_size
    self._dropout_rate = dropout_rate
    self._init_scale = init_scale
    self._final_init_scale = init_scale * final_init_scale_multiplier
    self._use_final_bias = use_final_bias
    self._activation = activation
    self._output_channels = output_channels

  def __call__(self, x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
    d_model = x.shape[-1]
    ffw_layer = hk.Linear(
        self._ffw_hidden_size,
        w_init=hk.initializers.VarianceScaling(self._init_scale))
    x = ffw_layer(x)

    x = self._activation(x)
    if is_training:
      x = hk.dropout(hk.next_rng_key(), self._dropout_rate, x)
    final_layer = hk.Linear(
        self._output_channels or d_model,
        w_init=hk.initializers.VarianceScaling(self._final_init_scale),
        with_bias=self._use_final_bias)
    return final_layer(x)


class SpaceTimeMultiHeadAttention(hk.Module):
  """SpaceTimeMultihead attention module with memory."""

  def __init__(
      self,
      *,
      value_size: int,
      key_size: int,
      num_heads: int,
      init_scale: float,
      dropout_rate: float,
      use_bias: bool,
      use_final_bias: bool,
      final_init_scale_multiplier: float,
      space_time_mode: str,
      name: str = 'spacetime_multihead_attention',
  ):
    """Initialises the SpaceTimeMultiHeadAttention module."""
    super().__init__(name=name)
    self._value_size = value_size
    self._key_size = key_size
    self._num_heads = num_heads
    self._dropout_rate = dropout_rate
    self._init_scale = init_scale
    self._final_init_scale = final_init_scale_multiplier * init_scale
    self._use_bias = use_bias
    self._use_final_bias = use_final_bias
    assert space_time_mode in ['space', 'time']
    self._space_time_mode = space_time_mode

  @hk.transparent
  def _multihead_linear(self, inputs: jnp.ndarray, hidden_size: int, name: str):
    linear = hk.Linear(
        self._num_heads * hidden_size,
        with_bias=self._use_bias,
        w_init=hk.initializers.VarianceScaling(scale=self._init_scale),
        name=name)
    out = linear(inputs)
    return jnp.reshape(out, inputs.shape[:-1] + (self._num_heads, hidden_size))

  @hk.transparent
  def _call_main(
      self,
      query: jnp.ndarray,
      key: jnp.ndarray,
      value: jnp.ndarray,
      is_training: bool,
  ) -> jnp.ndarray:
    batch_size, time_len, seq_len, embedding_size = query.shape

    query_heads = self._multihead_linear(query, self._key_size, 'query')
    key_heads = self._multihead_linear(key, self._key_size, 'key')
    value_heads = self._multihead_linear(value, self._value_size, 'value')
    if self._space_time_mode == 'time':
      einsum_template = 'btlhd,bTlhd->blhtT'
    else:
      einsum_template = 'btlhd,btLhd->bthlL'
    logits = jnp.einsum(einsum_template, query_heads, key_heads)

    scaled_logits = logits * self._key_size**(-0.5)
    weights = jax.nn.softmax(scaled_logits)

    if is_training:
      weights = hk.dropout(hk.next_rng_key(), self._dropout_rate, weights)
    if self._space_time_mode == 'time':
      einsum_template = 'blhtT,bTlhd->btlhd'
    else:
      einsum_template = 'bthlL,btLhd->btlhd'
    attn_vec = jnp.einsum(einsum_template, weights, value_heads)
    attn_vec = jnp.reshape(
        attn_vec, [batch_size, time_len,
                   seq_len, self._num_heads * self._value_size])

    final_linear = hk.Linear(
        embedding_size,
        w_init=hk.initializers.VarianceScaling(scale=self._final_init_scale),
        with_bias=self._use_final_bias)
    outputs = final_linear(attn_vec)

    return outputs

  def __call__(
      self,
      inputs: jnp.ndarray,
      is_training: bool,
  ) -> jnp.ndarray:
    """Computes the attention values.

    We use the following shape conventions: `B` for batch size, `T` for chunk
    size, and `D` for the embedding dimension.

    Args:
      inputs: array of shape [B, T, L, D]
      is_training: Whether to apply dropout

    Returns:
      An array of shape [B, T, D] the result of applying self-attention to
      inputs.
    """
    query = inputs
    key = value = inputs
    return self._call_main(
        query, key, value, is_training=is_training)


def spatio_temporal_gpt2_block(
    *,
    layer: int,
    mha_kwargs: Mapping[str, Any],
    ffw_kwargs: Mapping[str, Any],
    dropout_rate: float,
    is_training: bool,
    inputs: jnp.ndarray,
    use_layer_norm_bias: bool,
    final_residual: bool = True,
) -> jnp.ndarray:
  """Pure function for a single GPT-2 block."""
  time_attn = SpaceTimeMultiHeadAttention(name=f'h{layer}_time_attn',
                                          space_time_mode='time',
                                          **mha_kwargs)
  space_attn = SpaceTimeMultiHeadAttention(name=f'h{layer}_space_attn',
                                           space_time_mode='space',
                                           **mha_kwargs)
  dense_block = DenseBlock(name=f'h{layer}_mlp', **ffw_kwargs)
  ln_time = hk.LayerNorm(
      axis=-1,
      create_scale=True,
      create_offset=use_layer_norm_bias,
      name=f'h{layer}_ln_time')
  ln_space = hk.LayerNorm(
      axis=-1,
      create_scale=True,
      create_offset=use_layer_norm_bias,
      name=f'h{layer}_ln_space')
  attn_input = ln_time(inputs)
  h_attention = time_attn(
      inputs=attn_input,
      is_training=is_training)
  if is_training:
    h_attention = hk.dropout(hk.next_rng_key(), dropout_rate, h_attention)
  h = inputs + h_attention

  attn_input = ln_space(h)
  h_attention = space_attn(
      inputs=attn_input,
      is_training=is_training)
  if is_training:
    h_attention = hk.dropout(hk.next_rng_key(), dropout_rate, h_attention)
  h = h + h_attention

  h_ffw = dense_block(
      _layer_norm(
          h, name=f'h{layer}_ln_2', use_bias=use_layer_norm_bias),
      is_training)

  if is_training:
    h_ffw = hk.dropout(hk.next_rng_key(), dropout_rate, h_ffw)

  if final_residual:
    return h + h_ffw
  else:
    return h_ffw


class SpatioTemporalTransformerXL(hk.Module):
  """TimeTransformer-XL implementation."""

  def __init__(
      self,
      d_model: int,
      num_layers: int,
      num_heads: int,
      key_size: int,
      value_size: int,
      ffw_hidden_size: int,
      dropout_rate: float,
      absolute_position_length: Tuple[int, int] = (0, 0),
      use_layer_norm_bias: bool = True,
      same_attention_length: bool = False,
      use_attn_bias: bool = False,
      remat: bool = False,
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu,
      name: str = 'transformer_xl',
  ):
    """Initialises the module.

    Args:
      d_model: Size of the embeddings.
      num_layers: Number of transformer block layers.
      num_heads: Number of attention heads to use.
      key_size: Size of key (and query) embedding for attention.
      value_size: Size of value embedding for attention.
      ffw_hidden_size: Hidden size for MLP that follows attention.
      dropout_rate: How much dropout to apply to attention and MLP modules.
        used if greater than 0 and relative_position_embeddings is True.
      absolute_position_length: How many tokens to embed using absolute position
        embeddings. Note that this can be toggled on or off independently of
        `relative_position_embeddings`. The default (`0`) indicates no absolute
        position embeddings.
      use_layer_norm_bias: Whether to use a learnable bias for layer norm.
      same_attention_length: Whether each token attends over the same history
        length.
      use_attn_bias: Whether or not to use biases in attention linear layers.
      remat: Whether to use gradient rematerialization.
      activation: The nonlinearity to use in the DenseBlocks.
      name: The Haiku name of the module.
    """
    super().__init__(name=name)
    self._d_model = d_model
    self._num_layers = num_layers
    self._dropout_rate = dropout_rate
    self._remat = remat
    self._absolute_position_length = absolute_position_length
    self._use_layer_norm_bias = use_layer_norm_bias
    self._same_attention_length = same_attention_length

    self._mha_kwargs = dict(
        value_size=value_size,
        key_size=key_size,
        num_heads=num_heads,
        init_scale=2. / np.sqrt(self._num_layers),
        dropout_rate=self._dropout_rate,
        use_bias=use_attn_bias,
        use_final_bias=True,
        final_init_scale_multiplier=1.,
    )
    self._ffw_kwargs = dict(
        ffw_hidden_size=ffw_hidden_size,
        dropout_rate=self._dropout_rate,
        init_scale=2. / np.sqrt(self._num_layers),
        final_init_scale_multiplier=1.,
        use_final_bias=True,
        activation=activation,
    )

  def _get_position_embedding(
      self,
      inputs: jnp.DeviceArray,
  ) -> jnp.DeviceArray:
    """Computes absolute positional embeddings.

    Args:
      inputs: input token embeddings array of shape [B, L, T, D]

    Returns:
      The absolute positional embeddings of shape [B, L, T, D].
    """
    _, l, t, d = inputs.shape
    position_embeddings = hk.get_parameter(
        name='position_embedding',
        shape=list(self._absolute_position_length) + [self._d_model],
        init=hk.initializers.TruncatedNormal(stddev=0.02))
    assert l <= self._absolute_position_length[0]
    assert t <= self._absolute_position_length[1]
    assert d == self._d_model
    return position_embeddings[:l, :t]

  def __call__(
      self,
      input_embeddings: jnp.ndarray,
      is_training: bool = True,
  ) -> jnp.ndarray:
    """Computes the logits and next memory.

    Args:
      input_embeddings: array of shape [B, T, d_model]
      is_training: Whether to use dropout.

    Returns:
      The final layer embeddings
    """
    assert len(input_embeddings.shape) == 4
    assert len(self._absolute_position_length) == 2
    assert self._absolute_position_length[0] > 0
    assert self._absolute_position_length[1] > 0
    input_embeddings += self._get_position_embedding(input_embeddings)

    h = input_embeddings
    if is_training:
      h = hk.dropout(hk.next_rng_key(), self._dropout_rate, h)

    for i in range(self._num_layers):
      # Parameterize function on options.
      block = functools.partial(
          spatio_temporal_gpt2_block,
          layer=i,
          mha_kwargs=self._mha_kwargs,
          ffw_kwargs=self._ffw_kwargs,
          dropout_rate=self._dropout_rate,
          use_layer_norm_bias=self._use_layer_norm_bias,
          is_training=is_training)
      # Optionally rematerialize at the block level
      if self._remat:
        block = hk.remat(block)
      h = block(inputs=h)

    h = _layer_norm(h, name='ln_f', use_bias=self._use_layer_norm_bias)

    return h
