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
    def create_trainer(self, model, max_steps=60):
        self.trainer = SFTTrainer(
            model = model.model,
            tokenizer = model.tokenizer,
            train_dataset = self.dataset,
            dataset_text_field = "text",
            max_seq_length = model.max_seq_length,
            dataset_num_proc = 2,
            packing = False, # Can make training 5x faster for short sequences.
            args = TrainingArguments(
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
            ),
        )

class TestModel():
    def __init__(self, model_name):
        # TODO: These three are duplicated with trainer
        self.max_seq_length = 2048
        self.dtype = None
        self.load_in_4bit = True
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name, # YOUR MODEL YOU USED FOR TRAINING
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
        from transformers import model_namer
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
    model = Model(model_name = "unsloth/llama-3-8b-bnb-4bit", inference = False)
    model.model_to_peft_model()
    trainer = Trainer()
    trainer.load_data("vicgalle/alpaca-gpt4")
    trainer.convert_dataset_to_sharegpt(conversation_extension=3)
    trainer.standardize_dataset()
    trainer.set_chat_template(model)
    trainer.create_trainer(model, max_steps=60)
    trainer_stats = trainer.trainer.train()
    # trainer.run_inference()
    model.save_lora("TEST_LORA")
    model.save_model("TEST_MODEL")

    if False:
        lora_model = TestModel()

    if False:
        test = TestAutoModel()




