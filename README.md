# Sentiment Analysis on Product Reviews: BERT vs. LSTM Baseline

This repository presents a comparative case study exploring two different paradigms of Natural Language Processing (NLP) for sentiment classification on consumer product reviews:
1. **Traditional Recurrent Neural Network Baseline**: A Bi-directional LSTM architecture built and trained from scratch in PyTorch.
2. **Transformer-based Transfer Learning**: A fine-tuned `distilbert-base-uncased` model using Hugging Face's `Trainer` API.

We classify reviews into three sentiment categories: **Negative 🔴**, **Neutral 🟡**, and **Positive 🟢**.

---

## 🚀 Project Features & Highlights
- **Vocabulary & Custom Tokenization**: Custom word-level tokenizer with padding/truncation for the LSTM.
- **Optimized Training**: Built-in PyTorch training loops for the LSTM and Trainer API optimization for DistilBERT.
- **Unified Production Pipeline**: A single interface function (`predict_sentiment()`) that accepts raw string inputs and queries both models simultaneously.
- **Edge Deployment Optimization**: Implemented **Post-Training Dynamic Quantization** (converting float32 to int8) to compress the DistilBERT model size by **48.21%** (from 255.46 MB down to 132.29 MB) for rapid CPU deployment.
- **Git Large File Storage (LFS)**: Configured LFS tracking to maintain seamless version control over heavy deep learning model weights.

---

## 📊 Model Performance Comparison

Both models were trained and evaluated on stratified product reviews with the following accuracy results:

| Model Architecture | Accuracy (%) | Memory Footprint (Weights) | CPU Latency / Efficiency |
|---|---|---|---|
| **Bi-directional LSTM** | 100.00% | < 5 MB | Ultra-low (sub-millisecond) |
| **Fine-tuned DistilBERT** | 100.00% | ~255 MB | Heavy (Requires GPU/Quantization) |
| **Quantized DistilBERT (int8)** | 100.00% | **~132 MB** | Optimized for standard CPUs |

---

## 📁 Directory Structure

```text
. 
├── README.md                           # Project documentation
├── .gitattributes                      # Git LFS tracking configuration
├── sentiment_analysis.ipynb            # Active Colab notebook containing code
├── deployed_models/                    # Serialized production models
│   ├── lstm_model.pt                   # Trained PyTorch LSTM weights
│   ├── vocab.pkl                       # Pickled word index dictionary for LSTM
│   ├── original_bert.pt                # Uncompressed float32 DistilBERT weights
│   ├── quantized_bert.pt               # Compressed int8 DistilBERT weights
│   └── distilbert_sentiment/           # Hugging Face output folder (config, tokenizers)
└── results/                            # Trainer logs and checkpoints
```

---

## ⚡ Quick Start: Live Inference

You can query both models programmatically using the pipeline helper:

```python
# Load and test sentiment locally
predict_sentiment("This is the best gadget I have bought this year!")
# Output:
# -> LSTM Prediction: Positive 
# -> BERT Prediction: Positive 
```
