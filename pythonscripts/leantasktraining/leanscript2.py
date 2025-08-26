from unsloth import FastLanguageModel # FastLanguageModel for LLMs
import torch
from datasets import load_dataset
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers.generation.streamers import TextStreamer
from datasets import Dataset
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from transformers import EarlyStoppingCallback
import os
import sklearn
import pandas as pd
os.environ["WANDB_PROJECT"]="LEANLLAMA3.1" # Set your project name for WandB
max_seq_length = 1500 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


def load_model():
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
alpaca_prompt = """You are a lean4 mathematical proof assistant. Given the following natural language problem, generate a formal statement of the natural language problem and generate mutliple formal proofs in the form of tactics in lean4 for the natural language problem in lean4. if the natural language answer is provided, use it to help you generate the formal proofs. If not, generate the formal proof without the natural language answer. Ensure that the proof is mathematically complete and that there are no errors and inconsistencies in the lean4 code.  
### Natural Language Statement:
{}

### Natural Language Answer:
{}

### Formal Statement:
{}

### Formal Proof:
{}"""
def formatting_prompts_func(examples):
    nls = examples["natural_language_statement"]
    ans = examples["answer"]
    fs = examples["formal_statement"]
    proofs = examples["proof"]
    texts = []
    for nl, answer, f, proof in zip(nls, ans, fs, proofs):
        answer_str = answer if answer and str(answer).strip() else ""
        f_str = f if f and str(f).strip() else ""
        if isinstance(proof, list):
            proof = "\n".join(proof)
        proof_str = proof if proof and str(proof).strip() else ""
        prompt = alpaca_prompt.format(nl, answer_str, f_str, proof_str)
        full_text = prompt + EOS_TOKEN
        texts.append(full_text)
    return {"text": texts}

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
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 2,
        warmup_steps = 100,
        num_train_epochs = 10, # Set this for 1 full training run.
        learning_rate = 1e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        eval_strategy="steps",
        eval_steps=200,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,        
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "LEANLLAMA3.1",
        report_to = "wandb",
        run_name="run1"  # Use this for WandB etc
    ),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]   
)
       
    return trainer

def main():
    model, tokenizer = load_model()
    global EOS_TOKEN
    EOS_TOKEN = tokenizer.eos_token 
   
    from datasets import Dataset
    from sklearn.model_selection import train_test_split
    import pandas as pd

    df= pd.read_json("data/lean_workbook.json")
    df = df[df['proof'].apply(lambda x: not isinstance(x, list) or len(x) > 0)].reset_index(drop=True)
    df["answer"] = df["answer"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    df["proof"] = df["proof"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)

    df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42)

# Split temp into validation (10%) and test (10%)
    df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
    df_train.head()
    dataset = Dataset.from_pandas(df_train)
    dataset = dataset.map(formatting_prompts_func, batched = True,)
    eval_dataset = Dataset.from_pandas(df_valid)
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched = True,)
    trainer = init_trainer(model, tokenizer, dataset,eval_dataset)
    trainer_stats = trainer.train(resume_from_checkpoint="LEANLLAMA3.1/checkpoint-1000")
    model.save_pretrained("trainedmodels/LL1") 
    tokenizer.save_pretrained("trainedmodels/LL1")
    

    
    
    
if __name__ == "__main__":
    main()