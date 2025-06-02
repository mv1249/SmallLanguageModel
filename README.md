# 🧠 Small Language Model (SLM) from Scratch

This project demonstrates how to build and train a **Small Language Model (SLM)** using PyTorch, inspired by GPT-style transformers. It includes everything from data preprocessing and tokenization to training and generation.

---

## 📌 Key Concepts Covered

- What is a Small Language Model (SLM)?
- Differences between SLMs and LLMs
- Tokenization methods (BPE, subword, etc.)
- Transformer architecture (Self-Attention, MLP, LayerNorm)
- Training from scratch with PyTorch
- Mixed precision training using `torch.amp`
- Auto-regressive text generation

---

## 🛠 Model Architecture

The architecture includes:
- Token + Positional Embedding Layer
- Stack of Transformer Blocks:
  - LayerNorm
  - Causal Self-Attention
  - Feedforward MLP
  - Residual connections
- Final LayerNorm
- Output projection head (vocab logits)

---

## 🧪 Training Pipeline

1. **Data Preparation**
   - TinyStories dataset is tokenized using BPE.
   - Dataset is chunked into fixed-size sequences for input/output pairs.

2. **Model Setup**
   - Configurable embedding size, number of layers, attention heads, etc.
   - Modular Transformer block with custom LayerNorm and attention logic.

3. **Optimizer & Scheduler**
   - `AdamW` with weight decay and tuned betas
   - Warmup + Cosine learning rate scheduler using `SequentialLR`

4. **Precision & Performance**
   - Supports mixed precision training (`float16` or `bfloat16`)
   - `torch.cuda.amp.GradScaler` used for stability

5. **Loss Estimation**
   - `estimate_loss()` evaluates both train and validation loss using inference mode

6. **Text Generation**
   - Model generates text autoregressively using top-k and temperature sampling

---

## 🚀 Getting Started

```bash
pip install torch
```

Train the model (pseudocode):
```python
model = GPT(GPTConfig(...))
optimizer = ...
scheduler = ...

for iter in range(max_iters):
    X, Y = get_batch('train')
    with ctx:
        logits, loss = model(X, Y)
        scaler.scale(loss).backward()
        ...  # optimizer step, zero_grad, scheduler step
```

Generate text:
```python
model.eval()
context = torch.tensor([[tokenizer.encode("Once upon a time")]])
output = model.generate(context, max_new_tokens=100)
print(tokenizer.decode(output[0]))
```

---

## 📁 Project Structure

- `Config`: Hyperparameters for model and training
- `Block`: Transformer block (LayerNorm, SelfAttention, MLP)
- `GPT`: Full model
- `train.py`: Training loop logic
- `utils.py`: Tokenization, batching, loss estimation

---

## 📚 References

- GPT-2 Paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- TinyStories Dataset: https://huggingface.co/datasets/roneneldan/TinyStories
- PyTorch AMP: https://pytorch.org/docs/stable/amp.html

---

## 🙌 Acknowledgements

Thanks to Raj and Team Vizura!

---
