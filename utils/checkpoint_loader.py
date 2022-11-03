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

"""A checkpoint loader."""
import pickle
from typing import Any, Mapping


_SAVED_VALUES = [
    'params',
    'state',
    'config',
    'augm_params',
    'augm_state',
    'augm_config'
]


def load_params_state(model_path) -> Mapping[str, Any]:
  saved_params = pickle.load(model_path)
  for param in _SAVED_VALUES:
    assert param in saved_params.keys(), f'Checkpoint is missing key {param}.'

  return {k: saved_params[k] for k in _SAVED_VALUES}
