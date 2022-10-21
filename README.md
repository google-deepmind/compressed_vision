# Compressed Vision

This repo contains the code for the ACCV paper on [Compressed Vision](https://sites.google.com/corp/view/compressed-vision).

## Downloading models
In particular, we provide pre-trained models that
can be used to compress your own data.
We provide both models that can be used to perform augmentation or just
compression.

## Installation

First, install dependencies following these instructions:

1. Create a virtual env: `python3 -m venv ~/.venv/compressed_vision`
2. Switch to the virtual env: `source ~/.venv/compressed_vision/bin/activate`
3. Follow instructions for installing JAX on your platform:
   https://github.com/google/jax#installation
4. Install other dependencies: `pip install -f requirements.txt`

After installing dependencies, you can open the notebooks in the `colabs` directory
using Jupyter or Colab.
Our colabs assume that you are running from the
`compressed_vision` directory.


## Usage

A simple CoLAB showing how to load our models at
 different compression levels can be found in
 `./colabs/demo.ipynb`. It also includes an example
 showing how augmentation is applied to a compressed
 video.

 To use this CoLAB, you need to do the following:

 1. Download the `models`. You can either download compression models
 (denoted by [CR=*]) or an augmentation model that does compression and
 optionally an augmentation.
Download the model you are interested in and update `_BASE_PATH`, `_SAVE_PATH` in the CoLAB.
  - [CR=786](https://storage.cloud.google.com/dm_compressed_vision/models/compression/41748435_4_cr%3D786.pkl)
  - [CR=384](https://storage.cloud.google.com/dm_compressed_vision/models/compression/41877908_2_cr%3D384.pkl)
  - [CR=236](https://storage.cloud.google.com/dm_compressed_vision/models/compression/42071788_1_cr%3D236.pkl)
  - [CR=192](https://storage.cloud.google.com/dm_compressed_vision/models/compression/41861759_1_cr%3D192.pkl)
  - [Augmentation = Flip](https://storage.cloud.google.com/dm_compressed_vision/models/augmentations/35225852_7_augm%3Dflip.pkl)
 2. Download the videos  and update the `VIDEO_TO_TEST` path in the CoLAB.
  - [Video 1](https://storage.cloud.google.com/dm_compressed_vision/data/video2.gif)
  - [Video 2](https://storage.cloud.google.com/dm_compressed_vision/data/video6.gif)
  - [Video 3](https://storage.cloud.google.com/dm_compressed_vision/data/video7.gif)
  - [Video 4](https://storage.cloud.google.com/dm_compressed_vision/data/video8.gif)


## Citing this work

```
@inproceedings{wiles22,
 title={Compressed Vision for Efficient Video Understanding},
 autors={Wiles, Olivia and Carreira, Joao and Barr, Iain and Zisserman, Andrew and Malinowski, Mateusz},
 conference={ACCV},
 year=2022
 }
```

## License and disclaimer

Copyright 2022 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
