# Proving Mathematical Theorems with Large Language Models

This repository contains the code, experiments, and resources for the MSc Data Science dissertation **“Proving Mathematical Theorems with Large Language Models”**, completed at **Heriot-Watt University**.

The project investigates the use of Large Language Models (LLMs) for:
- Generating **informal mathematical proofs**
- **Autoformalising** informal proofs into Lean 4
- Iteratively **verifying and improving proofs** using an interactive theorem prover

---

## Project Overview

Mathematical proofs written by humans are typically informal and expressed in natural mathematical language. Formal proofs, however, require strict logical syntax and verification by theorem provers such as Lean.

This project explores whether LLMs can bridge this gap by:
1. Generating informal proofs from informal theorem statements
2. Converting informal proofs into formal Lean 4 proofs
3. Using Lean as a verifier to improve informal reasoning iteratively

---

## Objectives

The main objectives of this project are:

1. Fine-tune LLMs to generate informal mathematical proofs  
2. Train LLMs to autoformalise informal proofs into Lean 4  
3. Verify generated proofs using the Lean theorem prover  
4. Improve informal proofs using Lean’s feedback in an iterative loop  

---

## Models Used

The following Large Language Models were used and fine-tuned:

- **LLaMA 3.1**  
  https://ai.meta.com/llama/

- **Qwen 2.5-Math**  
  https://github.com/QwenLM/Qwen2.5-Math

Parameter-efficient fine-tuning was performed using **LoRA**, with training accelerated using **Unsloth**.

---

## Datasets

### Informal Proof Generation

- **Natural Proofs Dataset** 
  https://github.com/wellecks/naturalproofs

- **DEMI-MathAnalysis Dataset** 
  https://huggingface.co/datasets/chenxwh/DEMI-MathAnalysis

### Autoformalisation

- **Lean Workbook Dataset** 
   https://huggingface.co/datasets/internlm/Lean-Workbook
  
- **FIMO Dataset**   
  https://github.com/ictnlp/FIMO

---

## Tools and Technologies

- **Python** 

- **Hugging Face Transformers**  
  https://github.com/huggingface/transformers

- **Unsloth**  
  https://github.com/unslothai/unsloth

- **LoRA (Low-Rank Adaptation)**  

- **Lean 4**  
  https://leanprover.github.io/

- **mathlib4**  
  https://github.com/leanprover-community/mathlib4
  
- **GPT-4 / GPT-4o** (evaluation)
  https://openai.com/index/hello-gpt-4o/ 

---

## Methodology

The workflow consists of three main tasks:

1. Fine-tuning LLMs on informal mathematical proof datasets  
2. Training on the task of autoformalising informal proofs into Lean 4 syntax  
3. Creating an iterative framework that improves informal proofs using feedback from Lean  

Lean is used as an automated verifier, returning errors that are fed back into the model to refine the proof.

---

## Evaluation

Evaluation includes:
- Correctness and completeness of informal proofs
- Lean verification success rate
- Comparison with base (non-fine-tuned) models
- Performance on held-out test datasets

