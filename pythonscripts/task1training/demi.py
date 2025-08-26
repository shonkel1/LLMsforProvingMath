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
import os
os.environ["WANDB_PROJECT"]="demirun"
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
    r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 192,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
    return model, tokenizer
proof_prompt = """Below is a {} problem. Write a detailed proof that correctly proves the statement.

### Statement:
{}

### Proof:
{}"""

def formatting_prompts_func(examples):
    types = examples["ProblemType"]
    statements = examples["Problem"]
    proofs = examples["Solution"]

    texts = []
    for typ, statement, proof in zip(types, statements, proofs):
        text = proof_prompt.format(typ, statement, proof) + EOS_TOKEN
        texts.append(text)

    return { "text": texts }

def init_trainer(model, tokenizer, dataset):
    FastLanguageModel.for_training(model) # Enable for training!    
    trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset, # dataset -> tokenized_train_dataset
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        
        # Use num_train_epochs = 1, warmup_ratio for full training runs!
        warmup_steps = 5,
        max_steps = 300, # 60 -> 10

        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
        
    return trainer

def main():
    

    # model_name = "unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit"
    # model_name = "unsloth/llava-1.5-7b-hf-bnb-4bit"
    model, tokenizer = load_model("NaturalProofs-Llama3.1secondearlystop")
    global EOS_TOKEN 
    EOS_TOKEN = tokenizer.eos_token
    import pandas as pd
    
    df = pd.read_csv("/home/ssk2011/mathproofproj/pretraining_data (1).csv")

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(formatting_prompts_func, batched = True,)


    # generate_caption(dataset[0]["image"], model, tokenizer)
    
    trainer = init_trainer(model, tokenizer, dataset)

    trainer_stats = trainer.train()
    model.save_pretrained("NaturalProofs-Llama3.1+demiearlystop") 
    tokenizer.save_pretrained("NaturalProofs-Llama3.1+demiearlystop")
    
    
    
if __name__ == "__main__":
    main()
