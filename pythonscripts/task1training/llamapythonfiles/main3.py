from unsloth import FastLanguageModel # FastLanguageModel for LLMs
import torch
from datasets import load_dataset
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers.generation.streamers import TextStreamer
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from transformers import EarlyStoppingCallback
import os
os.environ["WANDB_PROJECT"]="NaturalproofsLlama"
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


def load_model(model_name: str):
    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
        "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
        "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
        "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

        "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
        "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model

        "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
        "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-VL-7B-Instruct",

        "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
        "unsloth/llava-1.5-7b-hf-bnb-4bit",
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    ] # More models at https://huggingface.co/unsloth

    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    model = FastLanguageModel.get_peft_model(
    model,
    r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
    return model, tokenizer



proof_prompt = """Below is a mathematical statement and along with context. Write a detailed and mathematically correct proof that correctly proves the statement.

### Statement:
{}

### Context:
{}

### Proof:
{}"""
def formatting_prompts_func(examples):
    titles = examples["title"]
    statements = examples["text"]    # The problem statement
    ctxs_list  = examples["ctxs"]    # List of dicts
    proofs     = examples["target"]  # The proof (target output)

    texts = []
    for p_title,statement, ctxs, proof in zip(titles, statements, ctxs_list, proofs):
        context_text = f"Problem Title: {p_title}\n\n"
        for ctx in ctxs:
            context_text += f"Title: {ctx.get('title', '')}\nContent: {ctx.get('text', '')}\n\n"

        text = proof_prompt.format(statement, context_text, proof) + EOS_TOKEN
        texts.append(text)

    return { "text": texts }

def init_trainer(model, tokenizer, dataset,eval_dataset):
    FastLanguageModel.for_training(model) # Enable for training!    
    trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset=eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 100,
        num_train_epochs = 10, # Set this for 1 full training run.
        learning_rate = 5e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        eval_strategy="steps",
        eval_steps=50,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        save_strategy="steps",
        save_steps=50,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "wandb",
        run_name="run3"  # Use this for WandB etc
    ),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]   
)
        
    return trainer

def main():
    

    # model_name = "unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit"
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    # model_name = "unsloth/llava-1.5-7b-hf-bnb-4bit"
    model, tokenizer = load_model(model_name)
    global EOS_TOKEN 
    EOS_TOKEN = tokenizer.eos_token
    import pandas as pd
    splits = {
    'test': 'naturalproofsfiles/naturalproofs_gen_test (1).jsonl',
    'train': 'naturalproofsfiles/naturalproofs_gen_train (1).jsonl',
    'valid': 'naturalproofsfiles/naturalproofs_gen_valid (1).jsonl'
    }

    df_train = pd.read_json(splits["train"], lines=True)
    df_valid = pd.read_json(splits["valid"], lines=True)
    df_test  = pd.read_json(splits["test"], lines=True)


    dataset = Dataset.from_pandas(df_train)
    dataset = dataset.map(formatting_prompts_func, batched = True,)
    eval_dataset = Dataset.from_pandas(df_valid)
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched = True,)
    trainer = init_trainer(model, tokenizer, dataset,eval_dataset)

    # generate_caption(dataset[0]["image"], model, tokenizer)
    
    

    trainer_stats = trainer.train()
    model.save_pretrained("NLearlystop3") 
    tokenizer.save_pretrained("NLearlystop3")
    
    
    
if __name__ == "__main__":
    main()