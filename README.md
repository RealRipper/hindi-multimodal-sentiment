# Hindi Multimodal News Sentiment Analysis

Replicating [Das & Singh (2022)](https://doi.org/10.1016/j.eswa.2022.117575) — a multi-stage multimodal sentiment framework — on Hindi news data.

## What this is
The original paper achieves 89.3% accuracy on Assamese news using text + image fusion.
This project adapts that architecture to Hindi, modernises the encoders (MuRIL, ResNet-50),
and measures the multimodal gain over a text-only baseline.

## Architecture
1. **Text branch** — CNN-LSTM on MuRIL embeddings
2. **Visual branch** — BLIP caption → Hindi translation → sentiment
3. **Intermediate fusion** — joint text + image features
4. **Decision fusion** — majority vote over all three branches

## Stack
- PyTorch, HuggingFace Transformers, MuRIL
- IndicSentiment (v1 dev data), Hindi news scrape (Phase 2)
- Kaggle (GPU), plain CSV experiment logs

## Repo layout
