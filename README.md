# News Topic Classification

A PyTorch-based machine learning pipeline for classifying news articles into four categories (World, Sports, Business, Sci/Tech). The goal was to build a clean, hand-written deep learning workflow covering text preprocessing, vocabulary building, custom datasets, and training loops — moving beyond basic scikit-learn models.

## Dataset

[AG News Dataset](https://paperswithcode.com/dataset/ag-news) (120,000 training samples, 7,600 test samples). The dataset is perfectly balanced with 30,000 training samples per class.

The data is downloaded directly as CSV files (title and description columns), which are combined during preprocessing.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download the dataset CSVs
python3 download_data.py
```

## Usage

Run the full pipeline (builds vocabulary, trains the LSTM model, evaluates, and saves artifacts):

```bash
python3 train_model.py
```

Run inference on a few sample text headlines:

```bash
python3 predict.py
```

## Results

| Model | Accuracy | F1 (macro) |
|---|---|---|
| Embedding + LSTM | 0.91 | 0.91 |

Per-class performance (F1-score):
- World: 0.91
- Sports: 0.96 (Easiest to classify due to specific terminology)
- Business: 0.87
- Sci/Tech: 0.89

The model struggles slightly with distinguishing between Business and Sci/Tech articles, as there is often overlap in vocabulary (e.g., tech company earnings reports). Sports articles are very distinct and achieve the highest accuracy.

Confusion matrix plots and class distribution graphs are saved to the `results/` directory.

## Project Structure

```
├── data/                    # Dataset CSVs (downloaded)
├── notebooks/
│   └── exploration.ipynb    # EDA and text analysis
├── src/
│   ├── dataset.py           # Preprocessing, Vocab, and PyTorch Dataset
│   ├── model.py             # LSTM Neural Network definition
│   ├── train.py             # Training and validation loops
│   └── evaluate.py          # Metrics and confusion matrix plotting
├── models/                  # Saved weights (.pt) and vocab (.json)
├── results/                 # Confusion matrix and EDA plots
├── train_model.py           # Main training entry point
├── predict.py               # Inference script
├── download_data.py         # Helper script to fetch data
└── requirements.txt
```

## What I'd improve next

- Experiment with pre-trained word embeddings (like GloVe or Word2Vec) instead of training the embedding layer from scratch
- Try a bidirectional LSTM or GRU architecture to capture context from both directions
- Add learning rate scheduling to fine-tune the final epochs
- Improve tokenization (e.g., handling specific punctuation, maybe subword tokenization)
