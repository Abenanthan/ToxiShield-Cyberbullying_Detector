# ToxiShield Cyberbullying Detector

A machine learning project for detecting cyberbullying in text comments using a TensorFlow-based deep learning model.

## Overview

This project analyzes textual comments and classifies them as:

- Bullying
- Not-Bullying

It includes:

- Dataset loading and exploration
- Text preprocessing and vectorization
- LSTM-based deep learning model training
- Evaluation and visualization
- A simple Gradio interface for testing predictions

## Project Files

- `FDS_PROJECT_CYBER_BULLYING_DETECTION.ipynb` — main notebook containing data exploration, model building, and training workflow
- `train1.csv` — training dataset with text comments and labels

## Dataset

The dataset contains comment text and labels, with columns similar to:

- `Text` — the text comment
- `Label` — classification label (`Bullying` or `Not-Bullying`)
- `Types` — bullying type/category when applicable

## Requirements

Install the following Python packages before running the notebook:

```bash
pip install pandas numpy matplotlib tensorflow gradio
```

Optional but useful for notebook environments:

```bash
pip install jupyter
```

## Setup

1. Open the project folder.
2. Start Jupyter Notebook or VS Code Notebook support.
3. Open `CYBER_BULLYING_DETECTION.ipynb`.
4. Run the cells in order.

## How It Works

1. Load the dataset from `train1.csv`.
2. Inspect the dataset and text distribution.
3. Transform text comments using `TextVectorization`.
4. Train a Bidirectional LSTM model.
5. Evaluate model accuracy and loss.
6. Optionally use the Gradio interface to test new comments.

## Model Architecture

The notebook uses a sequential neural network with:

- Text embedding layer
- Bidirectional LSTM layer
- Dense hidden layer
- Output layer for binary classification

## Notes

- The notebook appears to be designed for experimentation and educational use.
- Some cells are written for Google Colab-style file upload workflows.
- For local execution, you may need to update the dataset loading section if your CSV path differs.

## Example Run

```python
# Example for loading and checking the dataset
import pandas as pd

df = pd.read_csv('train1.csv')
print(df.head())
print(df['Label'].value_counts())
```

## Future Improvements

- Add more balanced data for better generalization
- Improve preprocessing with cleaning and stopword handling
- Explore transformer-based models such as BERT
- Add proper validation metrics and confusion matrix
- Deploy as a web app or API

## License

This project is intended for academic and learning purposes.
