# Transformers tutorial

## contents
| title | source code |
| --- | --- |
| Pipelines for inference | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nattyo1226/hf_tutorial/blob/main/out/pipelines.ipynb) |
| Load pretrained instances with an AutoClass | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nattyo1226/hf_tutorial/blob/main/out/autoclass.ipynb) |
| Preprocess | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nattyo1226/hf_tutorial/blob/main/out/preprocess.ipynb) |
| Fine-tune a pretrained model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nattyo1226/hf_tutorial/blob/main/out/finetune.ipynb) |

## how to setup environment
- local machine
    - `uv` (recommended)
        - `uv lock`: resolve dependency relations & create `uv.lock`
        - `make install`: install appropriate `PyTorch` & create `.venv`
    - `pip`
        - `pip install .`: install main dependencies
        - `pip install .[dev]`: install sub dependencies for developpment
- google colab
    - run first cell

## how to enjoy tutorials
- `src/*.py` (marimo botebook, recommended)
    - `(uv run) marimo edit path/to/src/*.py`
- `out/*.ipynb` (jupyter notebook)
    - open them in you favorite editor (vscode, jupyter, google colab ...)
