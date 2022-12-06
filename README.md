# Compressed Vision

This repo contains the code for the ACCV paper on [Compressed Vision](https://sites.google.com/corp/view/compressed-vision). The paper describes how we can first compress videos to
 a smaller representation and then train a neural network *directly* on this
 compressed representation for various downstream tasks. We can also apply
 augmentations directly in this compressed space, thereby replicating the full
 standard video pipeline. By operating directly
 in the compressed representation, we can save both memory and speed.

## Description
In particular, we provide pre-trained models that
can be used to compress and decompress your own data.
We provide both models that can be used to perform augmentation or just
compression.

## Installation

First, install dependencies following these instructions:

1. Create a virtual env: `python3 -m venv ~/.venv/compressed_vision` (we used
python==3.7.14)
2. Switch to the virtual env: `source ~/.venv/compressed_vision/bin/activate`
3. Follow instructions for installing JAX on your platform:
   https://github.com/google/jax#installation if you want to run on device
4. Install other dependencies: `pip install -r requirements.txt`

After installing dependencies, you can open the notebooks in the `colabs` directory
using Jupyter or Colab.
Our colabs assume that you are running from the
`compressed_vision` directory if running locally.


## Usage

A simple CoLAB showing how to load our models at
 different compression levels can be found in
 `./colabs/demo.ipynb`. It also includes an example
 showing how augmentation is applied to a compressed
 video.

Another simple CoLAB in `./colabs/demo_data.ipynb`
shows how to load compressed versions
and visualises the PCA of these representations as videos.

## Citing this work

```
@inproceedings{wiles22,
 title={Compressed Vision for Efficient Video Understanding},
 authors={Wiles, Olivia and Carreira, Joao and Barr, Iain and Zisserman, Andrew and Malinowski, Mateusz},
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
