import requests
import argparse
import json
import gc
from datetime import datetime
def generate_timestamp_name():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

from .train import Model, Trainer
from .datasetting import create_dataset


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



def run_dataset(model_name, max_steps, dataset, output, conversation_extension, num_train_epochs):
    model = Model(model_name = model_name, inference = False)
    model.model_to_peft_model()
    model.print_special_tokens()
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
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model", type=str, default="unsloth/llama-3-8b-bnb-4bit", help="Model name to use for training. Default is model_name='unsloth/llama-3-8b-bnb-4bit'")
    train_parser.add_argument("--max_steps", type=int, default=30, help="Max training steps. Default is max_steps=30")
    train_parser.add_argument("--dataset", type=str, default="vicgalle/alpaca-gpt4", help="Dataset to use for training. Default is dataset='vicgalle/alpaca-gpt4'")
    train_parser.add_argument("--output", default=None, help="Output file to save model to. Default is None")
    train_parser.add_argument("--conversation_extension", default=3, type=int, help="Conversation extension. Default is 3")
    train_parser.add_argument("--num_train_epochs", default=None, type=int, help="Number of training epochs. Default is None")
    train_parser.add_argument("--ollama", action="store_true", help="Run Ollama to interact with the model. Default is False")

    dataset_parser = subparsers.add_parser("dataset", help="Run the dataset")
    # TODO: Change default model to Instruct
    dataset_parser.add_argument("--model", type=str, default="unsloth/llama-3-8b-bnb-4bit", help="Model name to use for training. Default is model_name='unsloth/llama-3-8b-bnb-4bit'")

    args = parser.parse_args()

    if args.command == "train":

        output_model_name = run_dataset(args.model, args.max_steps, args.dataset, args.output, args.conversation_extension, args.num_train_epochs)
        if args.ollama:
            ollama_interaction(output_model_name + "/Modelfile")

    elif args.command == "dataset":
        create_dataset(args.model)

    else:
        print("Invalid command. Please choose 'train' or 'dataset'")

if __name__ == "__main__":
    main()

