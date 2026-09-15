# ToxiShield — Cyberbullying & Toxic Comment Detector

A machine learning project for detecting toxic and harmful comments using a TensorFlow-based Bidirectional LSTM model, trained on the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge) dataset.

## Overview

This project analyzes textual comments and classifies them across six toxicity categories:

- **Toxic**
- **Severe Toxic**
- **Obscene**
- **Threat**
- **Insult**
- **Identity Hate**

A comment can belong to more than one category at once — this is a multi-label classification problem, not a single-class one.

It includes:

- Dataset loading and exploration
- Text preprocessing and vectorization
- Bidirectional LSTM deep learning model training
- Evaluation and visualization
- An interactive Gradio interface for single-comment and batch testing

## Project Files

- `ToxiShield_Toxicity_Detection.ipynb` — main notebook containing data exploration, model building, training, and the Gradio demo *(rename to match your actual notebook filename if different)*
- `train.csv` — training dataset with comments and their toxicity labels (download from the [Kaggle dataset page](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge) — not included in this repo due to size)

## Dataset

Sourced from the Jigsaw Toxic Comment Classification Challenge, containing Wikipedia talk-page comments labeled by human raters.

| Column | Description |
|---|---|
| `id` | Unique comment identifier |
| `comment_text` | The raw comment text |
| `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` | Binary (0/1) labels — a comment may have multiple labels set |

> **Note:** Only `train.csv` is used for training. `test.csv` and `test_labels.csv` are provided by Kaggle for competition scoring and are not required to run this notebook. See the [dataset page](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge) for the full file list.

## Requirements

Install the following Python packages before running the notebook:

```bash
pip install pandas numpy matplotlib tensorflow gradio
```

Optional, for local notebook environments:

```bash
pip install jupyter
```

## Setup

1. Download `train.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge) and place it in the project folder (or use `kagglehub` to fetch it directly — see notebook Cell 1).
2. Open the project folder.
3. Start Jupyter Notebook, VS Code, or Google Colab.
4. Open `ToxiShield_Toxicity_Detection.ipynb`.
5. Run the cells in order.

## How It Works

1. Load the dataset from `train.csv`.
2. Explore the dataset — class distribution across all six toxicity categories.
3. Transform comment text using `TextVectorization`.
4. Train a Bidirectional LSTM model with a 6-unit sigmoid output layer (multi-label classification).
5. Evaluate accuracy and loss, and plot training curves.
6. Score sample comments individually with `score_comment()`.
7. Launch the Gradio interface to test new comments interactively.

## Model Architecture

The notebook uses a sequential neural network with:

- Text embedding layer
- Bidirectional LSTM layer
- Dense hidden layer (ReLU)
- Output layer with 6 sigmoid units — one per toxicity category

Loss function: binary crossentropy. Optimizer: Adam.

## Gradio Demo

The final notebook cell launches a Gradio web interface with three tabs:

- **Single Comment** — enter one comment and get a status verdict plus a breakdown of scores across all six toxicity categories
- **Batch Analysis** — paste multiple comments (one per line) and get a summary verdict for each
- **Model Info** — architecture details, training configuration, and a score-interpretation reference table

Run the last notebook cell to launch it locally (`demo.launch()`). No data is stored or transmitted externally — everything runs in-session.

## Notes

- The notebook is written for Google Colab-style file upload workflows; for local execution, update the dataset loading cell to point to your local `train.csv` path.
- Class imbalance is significant — `severe_toxic`, `threat`, and `identity_hate` have far fewer examples than `toxic`, `obscene`, and `insult`. Accuracy alone can be misleading; consider per-class precision/recall if extending this project.

## Future Improvements

- Add class-weighting or resampling to address label imbalance
- Improve preprocessing with text cleaning and stopword handling
- Explore transformer-based models such as BERT
- Add per-class validation metrics (precision, recall, F1, confusion matrices)
- Deploy as a hosted web app or API

## License

This project is intended for academic and learning purposes. Dataset is released under CC0 by the original Kaggle uploader; underlying comment text is governed by Wikipedia's CC-BY-SA.