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

"""Tools for transforming the input data."""
import chex
import jax


def convert_im_to_codes(
    codec_encoder,
    codec_params,
    codec_state,
    images: chex.Array,
    is_return_quantized: bool = False,
):
  """Converts input images to codes."""
  codes, _ = codec_encoder(
      codec_params,
      codec_state,
      jax.random.PRNGKey(42),
      images,
      is_return_quantized=is_return_quantized,
  )
  return codes


def convert_codes_to_im(
    codec_decoder,
    codec_params,
    codec_state,
    codes: chex.Array,
    is_quantized: bool = False
):
  """Decode codes."""
  images, _ = codec_decoder(
      codec_params,
      codec_state,
      jax.random.PRNGKey(42),
      codes,
      is_quantized=is_quantized
  )
  return images


def encode_decode(
    codec_encoder,
    codec_decoder,
    codec_params,
    codec_state,
    inputs: chex.Array,
):
  """Encode and decode video."""
  codes = convert_im_to_codes(
      codec_encoder,
      codec_params,
      codec_state,
      inputs,
      is_return_quantized=False,
  )

  outputs = convert_codes_to_im(
      codec_decoder=codec_decoder,
      codec_params=codec_params,
      codec_state=codec_state,
      codes=codes,
      is_quantized=False)

  return outputs, codes
