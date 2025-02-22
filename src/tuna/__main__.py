import argparse
from datetime import datetime
import gc
from .datasetting import create_dataset
from .chat import ollama_interaction
from .train import Model, Trainer

def generate_timestamp_name():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def run_dataset(model_name, max_steps, dataset, output, conversation_extension, num_train_epochs):
    model = Model(model_name = model_name, inference = False)
    model.model_to_peft_model()
    model.print_special_tokens()
    model.set_chat_template()
    trainer = Trainer()
    trainer.load_data(dataset)
    trainer.convert_dataset_to_sharegpt(conversation_extension=3)
    trainer.standardize_dataset()
    trainer.apply_chat_template(model)
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

def main():
    parser = argparse.ArgumentParser(description="Run training")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model", type=str, default="unsloth/llama-3-8b-Instruct-bnb-4bit", help="Model name to use for training. Default is model_name='unsloth/llama-3-8b-bnb-4bit'")
    train_parser.add_argument("--max_steps", type=int, default=30, help="Max training steps. Default is max_steps=30")
    train_parser.add_argument("--dataset", type=str, default="virattt/financial-qa-10k", help="Dataset to use for training. Default is dataset='vicgalle/alpaca-gpt4'")
    train_parser.add_argument("--output", default=None, help="Output file to save model to. Default is None")
    train_parser.add_argument("--conversation_extension", default=3, type=int, help="Conversation extension. Default is 3")
    train_parser.add_argument("--num_train_epochs", default=None, type=int, help="Number of training epochs. Default is None")
    train_parser.add_argument("--ollama", action="store_true", help="Run Ollama to interact with the model. Default is False")

    dataset_parser = subparsers.add_parser("dataset", help="Run the dataset")
    dataset_parser.add_argument("--model", type=str, default="unsloth/llama-3-8b-Instruct-bnb-4bit", help="Model name to use for training. Default is model_name='unsloth/llama-3-8b-bnb-4bit'")
    dataset_parser.add_argument("--dataset", type=str, default="virattt/financial-qa-10k", help="Dataset to use for training. Default is dataset='viraattt/financial-qa-10k'")

    args = parser.parse_args()

    if args.command == "train":
        output_model_name = run_dataset(args.model, args.max_steps, args.dataset, args.output, args.conversation_extension, args.num_train_epochs)
        if args.ollama:
            ollama_interaction(output_model_name + "/Modelfile")

    elif args.command == "dataset":
        create_dataset(args.model, args.dataset)

    else:
        print("Invalid command. Please choose 'train' or 'dataset'")

if __name__ == "__main__":
    main()

