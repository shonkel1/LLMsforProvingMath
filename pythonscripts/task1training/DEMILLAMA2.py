from unsloth import FastLanguageModel # FastLanguageModel for LLMs
import torch
from datasets import load_dataset
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers.generation.streamers import TextStreamer
from datasets import Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import os
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
os.environ["WANDB_PROJECT"]="DEMILLAMADIFFPARA"
def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "trainedmodels/llamanaturalproofsmodels/NLearlystop2",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    return model, tokenizer
proof_prompt = """You are a mathematical assistant. Below is a mathematical problem. Write an accurate and detailed proof that correctly proves the statement and is mathematically consistent. Use the $\epsilon$-$\delta$ method (e.g., when proving limits or continuity) when necessary and ensure that you apply it correctly. Use the correct mathematical notation throughout the proof.
### Problem Type:
{}

### Statement:
{}

### Proof:"""
def make_formatting_prompts_func(tokenizer):
    def formatting_prompts_func(examples):
        types = examples["ProblemType"]
        statements = examples["Problem"]
        proofs = examples["Solution"]

        outputs = []
        for typ, statement, proof in zip(types, statements, proofs):
            prompt = proof_prompt.format(typ, statement)
            completion = proof.strip() + tokenizer.eos_token
            full_text = prompt + "\n" + completion
            outputs.append(full_text)
        return outputs  # ✅ Must return a list of strings
    return formatting_prompts_func




def init_trainer(model, tokenizer, dataset, formatting_p_func):
    FastLanguageModel.for_training(model) # Enable for training!    
    trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset, # dataset -> tokenized_train_dataset
    dataset_text_field = None,
    formatting_func= formatting_p_func, # formatting_func -> make_formatting_prompts_func(tokenizer)
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer = tokenizer,
        response_template = "### Proof:\n",
        instruction_template = None,
    ),
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2,
        
        # Use num_train_epochs = 1, warmup_ratio for full training runs!
        warmup_steps = 7,
        max_steps = 100, # 60 -> 10
        learning_rate = 1e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        report_to= "wandb",
        run_name="run2",  # Use this
    ),
)
        
    return trainer
def main():
    

    # model_name = "unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit"
    # model_name = "unsloth/llava-1.5-7b-hf-bnb-4bit"
    model, tokenizer = load_model()
    import pandas as pd
    
    df = pd.read_csv("data/pretraining_data (1).csv")

    dataset = Dataset.from_pandas(df)
    formatting_p_func = make_formatting_prompts_func(tokenizer)




    # generate_caption(dataset[0]["image"], model, tokenizer)
    
    trainer = init_trainer(model, tokenizer, dataset,formatting_p_func)

    trainer_stats = trainer.train()
    model.save_pretrained("DL2newpara") 
    tokenizer.save_pretrained("DL2newpara")
    
    
    
if __name__ == "__main__":
    main()