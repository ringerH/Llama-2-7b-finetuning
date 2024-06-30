# Fine-Tuned Llama 2 7B with Quantization and PEFT

## Overview

This repository contains the fine-tuned version of the Llama 2 7B model, specifically adapted for legal domain applications. The fine-tuning process involves quantization and Parameter-Efficient Fine-Tuning (PEFT), enabling the model to operate efficiently in resource-constrained environments while maintaining high performance. The adapted weights are stored in 4-bit precision, and the model is optimized using mixed-precision techniques.

[**Notebook**](https://colab.research.google.com/github/ringerH/Assignment/blob/main/4_bit_llama_2_7b.ipynb)  
[**Hugging Face Model**](https://huggingface.co/hillol7/Llama-2-7b-chat-hf-sharded-bf16-fine-tuned-adapters)

## Key Features

1. **Quantization**: The model weights are quantized to 4-bit precision using a normalized float format. This allows the model to be more memory-efficient, making it suitable for deployment in environments with limited resources. Double quantization is employed to ensure that training efficiency is maximized.

2. **Low-Rank Adaptation**: The model uses low-rank adaptation for the self-attention projections. This involves factorizing these projections into two low-rank matrices, which reduces the computational complexity of linear transformations. Specifically, the Query, Key, Value, and Output projections are treated as low-rank matrices, facilitating more efficient mathematical operations.

3. **Mixed-Precision Training**: The model leverages mixed-precision training by combining half-precision (16-bit) and an 8-bit Adam optimizer. This approach balances computational efficiency and accuracy, reducing the overall resource requirements while maintaining the model’s performance.

## Fine-Tuning Process

The fine-tuning was carried out on a legal dataset to adapt the Llama 2 7B model for specialized tasks within the legal domain. The process involves the following steps:

1. **Data Preparation**: A curated legal dataset was prepared, ensuring diverse and comprehensive coverage of legal terminology and contexts.
2. **Quantization Setup**: The model weights were converted to 4-bit precision, applying double quantization for enhanced training efficiency.
3. **PEFT Implementation**: Low-rank adaptations were applied to the self-attention projections, facilitating efficient computation.
4. **Mixed-Precision Optimization**: The model was trained using a combination of half-precision and an 8-bit Adam optimizer to balance efficiency and performance.
5. **Validation and Testing**: The fine-tuned model was rigorously tested to ensure its effectiveness in legal tasks and its performance was benchmarked against baseline models.

## Usage

To use the fine-tuned model, you can load it directly from Hugging Face:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("hillol7/Llama-2-7b-chat-hf-sharded-bf16-fine-tuned-adapters")
model = AutoModelForCausalLM.from_pretrained("hillol7/Llama-2-7b-chat-hf-sharded-bf16-fine-tuned-adapters")

# Example input text
input_text = "Your legal query here..."
inputs = tokenizer(input_text, return_tensors="pt")

# Generate response
outputs = model.generate(**inputs)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
