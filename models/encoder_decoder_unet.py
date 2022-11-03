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

"""A Haiku compression model."""
from typing import Sequence, Tuple, Union

import chex
from compressed_vision.models import i3d
from compressed_vision.models import normalization
import haiku as hk
import jax
import jax.numpy as jnp


class CompressionConvEncoderDecoder(hk.Module):
  """Model for VQGAN style encoder/decoder (with a transformer)."""

  def __init__(self,
               num_channels: int = 3,
               bot_channels: int = 128,
               bot_channels_code_index: int = 32,
               stride=(1, 2, 2),
               kernel_shape=(1, 4, 4),
               num_conv_layers=1,
               res_bot_mult=4.0,
               extra_encode_conv_layers=2,
               extra_encode_conv_kernel=(1, 3, 3),
               extra_encode_conv_use_bn=False,
               extra_decode_conv_layers=2,
               extra_decode_conv_use_bn=False,
               extra_decode_conv_kernel=(1, 3, 3),
               use_multicodebook: bool = True,
               use_ratio_for_codebooks: bool = False,
               use_attention: bool = False,
               use_skips: bool = False,
               skips_one_embedding: bool = False,
               attention_resolutions: Sequence[int] = (),
               last_linearity: str = '',
               vq_embedding_dims: int = 16,
               vq_num_embeddings: int = 256,
               vq_commitment_cost: float = 0.9,
               vq_decay: float = 0.5,
               padding: str = 'VALID',
               use_normalized_embeddings: bool = False,
               use_tanh: bool = False,
               use_time_compression_2x: bool = False,
               name: str = 'CompressionConvEncoderDecoder'):
    """Initializes the module."""
    super().__init__(name=name)

    self._num_channels = num_channels

    self._bot_channels = bot_channels
    self._res_bot_mult = res_bot_mult
    self._stride = stride
    self._kernel_shape = kernel_shape
    self._num_conv_layers = num_conv_layers
    assert not use_attention
    self._attention_resolutions = attention_resolutions
    assert not use_skips
    assert not skips_one_embedding
    self._padding = padding
    self._use_tanh = use_tanh
    self._use_time_compression_2x = use_time_compression_2x

    # Used for the improved VQVAE setup from VIM.
    self._bot_channels_code_index = bot_channels_code_index
    assert not use_normalized_embeddings

    self._extra_encode_conv_kernel = extra_encode_conv_kernel
    self._extra_encode_conv_layers = extra_encode_conv_layers
    self._extra_encode_conv_use_bn = extra_encode_conv_use_bn

    self._extra_decode_conv_layers = extra_decode_conv_layers
    self._extra_decode_conv_kernel = extra_decode_conv_kernel
    self._extra_decode_conv_use_bn = extra_decode_conv_use_bn

    self._activation_fn = jax.nn.relu
    self._vq_embedding_dims = vq_embedding_dims
    self._vq_num_embeddings = vq_num_embeddings
    self._vq_commitment_cost = vq_commitment_cost
    self._vq_decay = vq_decay

    self._conv_layer_channels = [
        bot_channels // pow(2, self._num_conv_layers - i - 1)
        for i in range(self._num_conv_layers)
    ]

    if use_multicodebook:
      self._num_codebooks, remainder = divmod(bot_channels, vq_embedding_dims)
      assert not remainder, remainder
    else:
      self._num_codebooks = 1
      self._vq_embedding_dims = bot_channels

    if last_linearity:
      self._final_linearity = getattr(jax.nn, last_linearity)
    else:
      self._final_linearity = lambda x: x

    vqvae_module = hk.nets.VectorQuantizerEMA

    # Initialised here so they can be accessed by multiple methods.
    self._all_vqs = []
    self._all_codebooks = []
    vqs = []
    for i in range(self._num_codebooks):
      vqs.append(
          vqvae_module(
              embedding_dim=self._vq_embedding_dims,
              num_embeddings=self._vq_num_embeddings,
              commitment_cost=self._vq_commitment_cost,
              decay=self._vq_decay,
              name=f'vqvae_{i}'))
    self._all_vqs.append(vqs)
    self._all_codebooks.append(self._num_codebooks)

    im_size = 32
    for idx, num_channels in enumerate(self._conv_layer_channels, 1):
      vqs = []
      ratio = (im_size // 8) ** 2
      if use_ratio_for_codebooks:
        num_codebooks = (self._num_codebooks // ratio)
      else:
        num_codebooks = self._num_codebooks

      t_embedding_dim = num_channels // num_codebooks
      im_size = im_size // 2
      num_embeddings = self._vq_num_embeddings

      for i in range(num_codebooks):
        vqs.append(
            vqvae_module(
                embedding_dim=t_embedding_dim,
                num_embeddings=num_embeddings,
                commitment_cost=self._vq_commitment_cost,
                decay=self._vq_decay,
                name=f'vqvae_{idx}_{i}'))

      self._all_vqs.append(vqs)
      self._all_codebooks.append(num_codebooks)

  def encode_embedding(
      self,
      inputs: chex.Array,
      is_training: bool,
  ) -> chex.Array:
    """Encodes input."""
    x = inputs  # Expected shape  [B, T, H, W, C].
    chex.assert_rank(x, 5)
    if x.shape[-1] != self._num_channels:
      raise ValueError(f'Input shape: {x.shape} does not match the num_channels'
                       f' passed to the constructor: {self._num_channels}.')

    for i, num_channels in enumerate(self._conv_layer_channels):
      if i == self._num_conv_layers - 1:
        activation = None
      else:
        activation = self._activation_fn

      x = i3d.Unit3D(
          output_channels=num_channels,
          kernel_shape=self._kernel_shape,
          stride=self._stride,
          activation_fn=activation,
          normalize_fn=None,
          padding=self._padding,
          name=f'conv_{i}')(x, is_training=is_training)

    if self._use_time_compression_2x:
      x = i3d.Unit3D(
          output_channels=self._conv_layer_channels[-1],
          kernel_shape=self._kernel_shape,
          stride=(2, 1, 1),
          activation_fn=activation,
          normalize_fn=None,
          padding=self._padding,
          name='conv_temporal')(x, is_training=is_training)

    if self._extra_encode_conv_use_bn:
      norm_fn = normalization.get_normalize_fn('batch_norm')
    else:
      norm_fn = None

    for i in range(self._extra_encode_conv_layers):
      x_new = self._activation_fn(x)

      x_new = i3d.Unit3D(
          output_channels=int(self._bot_channels*self._res_bot_mult),
          kernel_shape=self._extra_encode_conv_kernel,
          stride=(1, 1, 1),
          activation_fn=self._activation_fn,
          normalize_fn=norm_fn,
          padding=self._padding,
          name=f'encoder_res_0_{i}')(x_new, is_training=is_training)

      x_new = i3d.Unit3D(
          output_channels=self._bot_channels,
          kernel_shape=(1, 1, 1),
          stride=(1, 1, 1),
          activation_fn=None,
          normalize_fn=norm_fn,
          padding=self._padding,
          name=f'encoder_res_1_{i}')(x_new, is_training=is_training)

      x = x_new + x

    if self._use_tanh:
      x = jax.nn.tanh(x)

    assert x.shape[-1] % self._vq_embedding_dims == 0, x.shape
    return x

  def decode_embedding(self,
                       inputs: chex.Array,
                       is_training: bool) -> chex.Array:
    """Decodes quantized embedding to logits."""
    x = inputs  # Assumed shape [B, T, H, W, C]
    chex.assert_rank(x, 5)

    if self._extra_encode_conv_use_bn:
      norm_fn = normalization.get_normalize_fn('batch_norm')
    else:
      norm_fn = None

    for i in range(self._extra_decode_conv_layers):
      x_inputs = x
      x = self._activation_fn(x)

      x = i3d.Unit3D(
          output_channels=int(self._bot_channels * self._res_bot_mult),
          kernel_shape=self._extra_decode_conv_kernel,
          stride=(1, 1, 1),
          activation_fn=self._activation_fn,
          normalize_fn=norm_fn,
          padding=self._padding,
          name=f'decoder_res_0_{i}')(x, is_training=is_training)

      x = i3d.Unit3D(
          output_channels=self._bot_channels,
          kernel_shape=(1, 1, 1),
          stride=(1, 1, 1),
          activation_fn=None,
          normalize_fn=norm_fn,
          padding=self._padding,
          name=f'decoder_res_1_{i}')(x, is_training=is_training)

      x = x + x_inputs

    if self._use_time_compression_2x:
      x = hk.Conv3DTranspose(
          self._conv_layer_channels[-1],
          self._kernel_shape,
          (2, 1, 1),
          # target_shape,
          name='decoder_temporal')(x)

    output_channel_order = reversed(
        [self._num_channels] + self._conv_layer_channels[:-1])
    for i, num_channels in enumerate(output_channel_order):
      _, time, height, width, _ = x.shape
      target_shape = [stride*dim_size for stride, dim_size
                      in zip(self._stride, (time, height, width))]

      x = hk.Conv3DTranspose(
          num_channels,
          self._kernel_shape,
          self._stride,
          target_shape,
          name=f'decoder_{i}')(x)

      if i < len(self._conv_layer_channels) - 1:  # No activation if last layer.
        x = self._activation_fn(x)

    return self._final_linearity(x)

  def encode(
      self,
      inputs: chex.Array,
      is_return_quantized: bool = False,
  ) -> Union[chex.Array, Tuple[chex.Array, chex.Array]]:
    """Encodes to codes."""
    embeddings = self.encode_embedding(
        inputs,
        is_training=False,
    )
    _, codes, _ = self.quantize(embeddings, is_training=False)

    if is_return_quantized:
      quantized = self.codes_to_quantize(codes)
      return codes, quantized
    else:
      return codes

  def codes_to_quantize(
      self,
      codes: chex.Array,
  ) -> chex.Array:
    """Encodes to quantize."""
    chex.assert_rank(codes, 5)
    quantized = [vq.quantize(codes[..., i])
                 for i, vq in enumerate(self._all_vqs[0])]
    quantized = jnp.concatenate(quantized, axis=-1)

    return quantized

  def decode(
      self,
      inputs: Tuple[chex.Array, Sequence[chex.Array]],
      is_quantized: bool = False
  ) -> chex.Array:
    """Decodes from codes."""
    x = inputs  # Assumed shape [B, T, H, W, C]
    if is_quantized:
      quantized = x
    else:
      quantized = self.codes_to_quantize(codes=x,)
    return self.decode_embedding(
        quantized,
        is_training=False,
    )

  def quantize(
      self,
      inputs: chex.Array,
      is_training: bool,
      quantize_idx: int = 0,
  ) -> Tuple[chex.Array, chex.Array, chex.Array]:
    x = inputs  # Assumed shape [B, T, H, W, C]
    chex.assert_rank(x, 5)

    # [B, T, H, W, C] -> [B, T, H, W, C", D]
    batch_size, time_size, height, width, channels = x.shape
    x = x.reshape(
        batch_size, time_size, height, width,
        channels // self._all_codebooks[quantize_idx],
        self._all_codebooks[quantize_idx])

    vq_out = []
    for i, vq in enumerate(self._all_vqs[quantize_idx]):
      vq_out.append(vq(x[..., i], is_training))

    quantized, partial_loss, codes = [
        [out[value] for out in vq_out]
        for value in ['quantize', 'loss', 'encoding_indices']]

    quantized = jnp.concatenate(quantized, axis=4)
    codes = jnp.stack(codes, axis=4)
    loss = jnp.mean(jnp.asarray(partial_loss))

    return quantized, codes, loss

  def __call__(
      self,
      inputs: chex.Array,
      is_training: bool,
  ) -> Tuple[chex.Array, Sequence[chex.Array], chex.Array, chex.Array]:
    """Runs the full embed+quantise+reconstruct model."""
    embedding = self.encode_embedding(
        inputs,
        is_training,
    )

    quantized, codes, loss = self.quantize(embedding, is_training)

    reconstruction = self.decode_embedding(
        quantized,
        is_training,
    )

    return quantized, [codes], loss, reconstruction
