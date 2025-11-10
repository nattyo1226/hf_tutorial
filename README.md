# Transformers Tutorial

## Contents
| Title | Open in Colab |
| --- | --- |
| Pipelines for inference | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nattyo1226/hf_tutorial/blob/main/jupyter/pipelines.ipynb) |
| Load pretrained instances with an AutoClass | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nattyo1226/hf_tutorial/blob/main/jupyter/autoclass.ipynb) |
| Preprocess | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nattyo1226/hf_tutorial/blob/main/jupyter/preprocess.ipynb) |
| Fine-tune a pretrained model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nattyo1226/hf_tutorial/blob/main/jupyter/finetune.ipynb) |

## How to setup environment
- local machine
    - `uv` (recommended)
        - `uv lock`: resolve dependency relations & create `uv.lock`
        - `make install`: install appropriate `PyTorch` & create `.venv`
    - `pip`
        - `pip install .`: install main dependencies
        - `pip install .[dev]`: install sub dependencies for developpment
- google colab
    - run first cell

## How to enjoy tutorials
- `marimo/*.py` (marimo botebook, recommended)
    - `(uv run) marimo edit path/to/*.py`
- `jupyater/*.ipynb` (jupyter notebook)
    - open them in you favorite editor (vscode, jupyter, google colab ...)

## Issues
- When attempting to build the environment on Miyabi with `uv`, it throws an error due to compatibility issues with `torchcodec`. Since `torchcodec` is not used for fine-tuning, please remove it from the dependencies on `pyproject.toml`.
