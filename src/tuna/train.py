import torch
import json
from datasets import load_dataset
from unsloth import FastLanguageModel, to_sharegpt, standardize_sharegpt, apply_chat_template, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer, TrainingArguments


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
        # # TODO: Is this pad_token needed. Should a EOS be added
        # ### CUSTOM
        # print("Adding pad_token")
        # PAD_TOKEN = "<|pad|>"
        # self.tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
        # self.tokenizer.padding_side = "right"
        # self.model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=8)
        # #########

        if inference:
            self.set_model_for_inference()
        else:
            print(f"Loaded {self.model_name} for training")

    def print_special_tokens(self):
        print(f"BOS token: {self.tokenizer.bos_token}")
        print(f"PAD token: {self.tokenizer.pad_token}")
        print(f"EOS token: {self.tokenizer.eos_token}")

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

    def set_chat_template(self):
        # self.chat_template = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.
        #
        # ### Instruction:
        # {INPUT}
        #
        # ### Response:
        # {OUTPUT}"""

        self.tokenizer.chat_template = """{SYSTEM}

### Instruction:
{INPUT}

### Response:
{OUTPUT}"""

    def set_model_for_inference(self):
        if not self.inference:
            self.inference = True
            FastLanguageModel.for_inference(self.model)
            print(f"{self.model_name} set for inference")
        else:
            print("Model already set for inference")

    def run_inference(self, content):
        print("hello there this is a test")
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
            # self.dataset = load_dataset("json", data_files=data_path, split="train")
            self.dataset = load_dataset("json", data_files={"train": "train.jsonl", "validation": "val.jsonl", "test": "test.jsonl"})
            self.dataset = self.dataset["train"]
        else:
            self.dataset = load_dataset(data_path, split="train")

    def convert_dataset_to_sharegpt(self, conversation_extension=3):
        # self.dataset = to_sharegpt(
        #     self.dataset,
        #     merged_prompt="{instruction}[[\nYour input is:\n{input}]]",
        #     output_column_name="output",
        #     conversation_extension=3,  # Select more to handle longer conversations
        # )
        self.dataset = to_sharegpt(
            self.dataset,
            merged_prompt="{question}[[\nInformation:\n{context}]]",
            output_column_name="answer",
            conversation_extension=conversation_extension,  # Select more to handle longer conversations
        )
    def standardize_dataset(self):
        self.dataset = standardize_sharegpt(self.dataset)


    def apply_chat_template(self, model):
        self.dataset = apply_chat_template(
            self.dataset,
            tokenizer=model.tokenizer,
            chat_template=model.tokenizer.chat_template,
            default_system_message = "You are a helpful assistant", # [OPTIONAL]
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
            # dataset_kwargs={
            #     "add_special_tokens": False,
            #     "append_concat_token": False,
            # }
        )
        if num_train_epochs:
            print(f"Setting num_train_epochs to {num_train_epochs}")
            training_args.max_steps = -1
            training_args.num_train_epochs = num_train_epochs

        self.trainer = SFTTrainer(
            model = model.model,
            tokenizer = model.tokenizer,
            train_dataset = self.dataset,
            # train_dataset = self.dataset["train"],
            # eval_dataset=self.dataset["validation"],
            dataset_text_field = "text",
            max_seq_length = model.max_seq_length,
            dataset_num_proc = 2,
            packing = False, # Can make training 5x faster for short sequences.
            args = training_args,
        )

