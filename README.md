<picture>
  <source srcset="./res/dynamic_dark.webp" media="(prefers-color-scheme: dark)">
  <img src="./res/dynamic_light.webp" alt="ant's friendly logo" width=400>
</picture>

## Description
ant is a PyTorch-based Deep Learning framework. It is inspired by [nanoGPT](https://github.com/karpathy/nanoGPT), although it does not borrow code from it. It tries to keep being research-friendly, while integrating (way) more features, following modern best practices and being modular. Specifically, among others, ant supports:
1) The CIFAR-10, OpenWebText, FineWeb-Edu and [ClimbMix](https://huggingface.co/datasets/nvidia/ClimbMix) (SoTA) datasets
2) Any Hugging Face or TokenMonster (SoTA) tokenizer
3) ResNet, GPT2, Llama 2, nGPT, and OLMo 2 models
4) The PSGD, DistributedShampoo, AdEMAMix, SOAP, Muon and Scion optimizers
5) [μP](https://www.microsoft.com/en-us/research/blog/%C2%B5transfer-a-technique-for-hyperparameter-tuning-of-enormous-neural-networks/) for zero-shot hyperparameter transfer
6) Downstream evaluations (HellaSwag, ARC etc.) through lm_eval during training
7) Model summary (parameters and FLOPS) and computational graph visualization
8) Offline logging of gradients and weights in simple .dat files
9) Attention heatmaps
10) Plotting through PGFPlots
11) Pure PyTorch attention, FlashAttention, FlexAttention and cuDNN attention with RoPE, ALiBi and Sliding Window Attention (SWA)
12) Distributed Data Parallel, torch.compile and Automatic Mixed Precision

<div align="center">
  <em>Model summary, learning rate schedule and logging in the terminal.</em><br>
  <img src="./res/log.png" height="600px"><br>
</div>

<div align="center">
  <em>Computational graph of a Transformer with RoPE.</em><br>
  <img src="./res/graph.png" height="800px">
</div>

<div align="center">
  <em>Gradient and weight statistics of the Embeddings Table.</em><br>
  <img src="./res/stats.png" height="400px">
</div>

<div align="center">
  <em>Attention weights for a Transformer block with twelve heads.</em><br>
  <img src="./res/attention.png" height="400px">
</div>

## Getting Started
1)  Clone the repo:

        git clone https://github.com/gvlassis/ant.git


2)  Install [PyTorch](https://pytorch.org/get-started/locally/) and [FlashAttention](https://github.com/Dao-AILab/flash-attention).
3)  Install `requirements.txt`:

        pip install -r requirements.txt
