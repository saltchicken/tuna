import gc
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from unsloth import to_sharegpt
from unsloth import standardize_sharegpt
from unsloth import apply_chat_template
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
import requests
import json
from unsloth import is_bfloat16_supported
from transformers import TextStreamer
import argparse
from datetime import datetime
def generate_timestamp_name():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",             # Gemma 2.2x faster!
] # More models at https://huggingface.co/unsloth

class Model():
    def __init__(self, model_name, inference=False, max_seq_length=2048, dtype=None, load_in_4bit=True):
        self.model_name = model_name
        self.inference = False
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )
        if inference:
            self.set_model_for_inference()
        else:
            print(f"Loaded {self.model_name} for training")

    def model_to_peft_model(self):
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
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

    def set_model_for_inference(self):
        if not self.inference:
            self.inference = True
            FastLanguageModel.for_inference(self.model)
            print(f"{self.model_name} set for inference")
        else:
            print("Model already set for inference")

    def run_inference(self, content):
        if not self.inference:
            print(f"Model not loaded for inference")
            return
        messages = [                    # Change below!
            {"role": "user", "content": content},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")

        text_streamer = TextStreamer(self.tokenizer, skip_prompt = True)
        _ = self.model.generate(input_ids, streamer = text_streamer, max_new_tokens = 128, pad_token_id = self.tokenizer.eos_token_id)

    def save_lora(self, lora_name):
        self.model.save_pretrained(lora_name)
        self.tokenizer.save_pretrained(lora_name)

    def save_model(self, model_name):
        # Save to 8bit Q8_0
        if True: self.model.save_pretrained_gguf(model_name, self.tokenizer,)

        # Save to 16bit GGUF
        if False: self.model.save_pretrained_gguf(model_name, self.tokenizer, quantization_method = "f16")

        # Save to q4_k_m GGUF
        if False: self.model.save_pretrained_gguf(model_name, self.tokenizer, quantization_method = "q4_k_m")

        # quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],


class Trainer():
    def __init__(self):
        self.dataset = None
        self.chat_template = None
        self.trainer = None

    def load_data(self, data_path):
        if data_path.endswith("jsonl"):
            self.dataset = load_dataset("json", data_files=data_path, split="train")
        else:
            self.dataset = load_dataset(data_path, split="train")

    def convert_dataset_to_sharegpt(self, conversation_extension=3):
        self.dataset = to_sharegpt(
            self.dataset,
            merged_prompt="{instruction}[[\nYour input is:\n{input}]]",
            output_column_name="output",
            conversation_extension=3,  # Select more to handle longer conversations
        )
    def standardize_dataset(self):
        self.dataset = standardize_sharegpt(self.dataset)

    def set_chat_template(self, model):
        self.chat_template = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

        ### Instruction:
        {INPUT}

        ### Response:
        {OUTPUT}"""


        self.dataset = apply_chat_template(
            self.dataset,
            tokenizer=model.tokenizer,
            chat_template=self.chat_template,
            # default_system_message = "You are a helpful assistant", << [OPTIONAL]
        )
    def create_trainer(self, model, max_steps=60, num_train_epochs=None):
        training_args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = max_steps,
            # num_train_epochs = 1, # For longer training runs!
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Use this for WandB etc
        )
        if num_train_epochs:
            print(f"Setting num_train_epochs to {num_train_epochs}")
            training_args.max_steps = -1
            training_args.num_train_epochs = num_train_epochs

        self.trainer = SFTTrainer(
            model = model.model,
            tokenizer = model.tokenizer,
            train_dataset = self.dataset,
            dataset_text_field = "text",
            max_seq_length = model.max_seq_length,
            dataset_num_proc = 2,
            packing = False, # Can make training 5x faster for short sequences.
            args = training_args,
        )


def run_dataset(model_name, max_steps, dataset, output, conversation_extension, num_train_epochs):
    model = Model(model_name = model_name, inference = False)
    model.model_to_peft_model()
    trainer = Trainer()
    trainer.load_data(dataset)
    trainer.convert_dataset_to_sharegpt(conversation_extension=3)
    trainer.standardize_dataset()
    trainer.set_chat_template(model)
    trainer.create_trainer(model, max_steps=max_steps, num_train_epochs=num_train_epochs)
    trainer_stats = trainer.trainer.train()
    # trainer.run_inference()
    if not output:
        output = generate_timestamp_name()
    model.save_lora(output + "_lora")
    model_output_name = output + "_model"
    model.save_model(model_output_name)
    print("Deleting training model and trainer")
    del model
    del trainer
    gc.collect()
    return model_output_name

def ollama_interaction(model_file):
    import subprocess
    subprocess.run(["ollama", "create", "test", "-f", model_file], check=True)


    print("Ollama is running. Type /bye to exit.")
    while True:
        user_input = input(">> ")
        if user_input.strip().lower() == "/bye":
            break
        # query = query_ollama("test", user_input)
        # print(query)
        stream_ollama("test", user_input)

    subprocess.run(["ollama", "rm", "test"], check=True)

    print("Ollama session ended. Continuing with the script...")

def stream_ollama(model, prompt):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }

    with requests.post(url, headers=headers, data=json.dumps(data), stream=True) as response:
        for line in response.iter_lines():
            if line:
                print(json.loads(line)["response"], end="", flush=True)
        print("\n")

def query_ollama(model, prompt):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Error: {response.status_code}, {response.text}"

# Example usage
# model_name = "mistral"  # Change this to the model you have installed
# prompt_text = "Tell me a joke."
# response = query_ollama(model_name, prompt_text)
# print(response)




def main():
    parser = argparse.ArgumentParser(description="Run training")
    parser.add_argument("--model", type=str, default="unsloth/llama-3-8b-bnb-4bit", help="Model name to use for training. Default is model_name='unsloth/llama-3-8b-bnb-4bit'")
    parser.add_argument("--max_steps", type=int, default=30, help="Max training steps. Default is max_steps=30")
    parser.add_argument("--dataset", type=str, default="vicgalle/alpaca-gpt4", help="Dataset to use for training. Default is dataset='vicgalle/alpaca-gpt4'")
    parser.add_argument("--output", default=None, help="Output file to save model to. Default is None")
    parser.add_argument("--conversation_extension", default=3, type=int, help="Conversation extension. Default is 3")
    parser.add_argument("--num_train_epochs", default=None, type=int, help="Number of training epochs. Default is None")
    parser.add_argument("--ollama", action="store_true", help="Run Ollama to interact with the model. Default is False")
    args = parser.parse_args()

    output_model_name = run_dataset(args.model, args.max_steps, args.dataset, args.output, args.conversation_extension, args.num_train_epochs)
    if args.ollama:
        ollama_interaction(output_model_name)

if __name__ == "__main__":
    main()

