from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from unsloth import to_sharegpt
from unsloth import standardize_sharegpt
from unsloth import apply_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from transformers import TextStreamer


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

class Trainer():
    def __init__(self):
        self.max_seq_length = 2048
        self.dtype = None
        self.load_in_4bit = True
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.chat_template = None
        self.trainer = None

    def load_base_model(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = "unsloth/llama-3-8b-bnb-4bit",
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )
    def convert_base_model_to_FLM(self):
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

    def load_data(self):
        self.dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")

    def convert_dataset_to_sharegpt(self):
        self.dataset = to_sharegpt(
            self.dataset,
            merged_prompt="{instruction}[[\nYour input is:\n{input}]]",
            output_column_name="output",
            conversation_extension=3,  # Select more to handle longer conversations
        )
    def standardize_dataset(self):
        self.dataset = standardize_sharegpt(self.dataset)

    def set_chat_template(self):
        self.chat_template = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

        ### Instruction:
        {INPUT}

        ### Response:
        {OUTPUT}"""


        self.dataset = apply_chat_template(
            self.dataset,
            tokenizer=self.tokenizer,
            chat_template=self.chat_template,
            # default_system_message = "You are a helpful assistant", << [OPTIONAL]
        )
    def create_trainer(self):
        self.trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = self.dataset,
            dataset_text_field = "text",
            max_seq_length = self.max_seq_length,
            dataset_num_proc = 2,
            packing = False, # Can make training 5x faster for short sequences.
            args = TrainingArguments(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                max_steps = 60,
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
            ),
        )
    def run_inference(self):
        FastLanguageModel.for_inference(self.model) # Enable native 2x faster inference
        messages = [                    # Change below!
            {"role": "user", "content": "Continue the fibonacci sequence! Your input is 1, 1, 2, 3, 5, 8,"},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")

        text_streamer = TextStreamer(self.tokenizer, skip_prompt = True)
        _ = self.model.generate(input_ids, streamer = text_streamer, max_new_tokens = 128, pad_token_id = self.tokenizer.eos_token_id)

    def save_lora(self):
        self.model.save_pretrained("lora_model")
        self.tokenizer.save_pretrained("lora_model")

    def save_model(self):
        # Save to 8bit Q8_0
        if True: self.model.save_pretrained_gguf("model", self.tokenizer,)

        # Save to 16bit GGUF
        if False: self.model.save_pretrained_gguf("model", self.tokenizer, quantization_method = "f16")

        # Save to q4_k_m GGUF
        if False: self.model.save_pretrained_gguf("model", self.tokenizer, quantization_method = "q4_k_m")

        # quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],

class Model():
    def __init__(self):
        # TODO: These three are duplicated with trainer
        self.max_seq_length = 2048
        self.dtype = None
        self.load_in_4bit = True
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model) # Enable native 2x faster inference

        #TODO: This is duplicated from tainers run_inference

        messages = [                    # Change below!
            {"role": "user", "content": "Describe anything special about a sequence. Your input is 1, 1, 2, 3, 5, 8,"},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")

        text_streamer = TextStreamer(self.tokenizer, skip_prompt = True)
        _ = self.model.generate(input_ids, streamer = text_streamer, max_new_tokens = 128, pad_token_id = self.tokenizer.eos_token_id)

class TestAutoModel():
    def __init__(self):
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
        self.load_in_4bit = True
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            "lora_model", # YOUR MODEL YOU USED FOR TRAINING
            load_in_4bit = self.load_in_4bit,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("lora_model")
                #TODO: This is duplicated from tainers run_inference

        messages = [                    # Change below!
            {"role": "user", "content": "Describe anything special about a sequence. Your input is 1, 1, 2, 3, 5, 8,"},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")

        text_streamer = TextStreamer(self.tokenizer, skip_prompt = True)
        _ = self.model.generate(input_ids, streamer = text_streamer, max_new_tokens = 128, pad_token_id = self.tokenizer.eos_token_id)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.load_base_model()
    trainer.convert_base_model_to_FLM()
    trainer.load_data()
    trainer.convert_dataset_to_sharegpt()
    trainer.standardize_dataset()
    trainer.set_chat_template()
    trainer.create_trainer()
    trainer_stats = trainer.trainer.train()
    trainer.run_inference()
    trainer.save_lora()
    trainer.save_model()

    if False:
        lora_model = Model()

    if False:
        test = TestAutoModel()




