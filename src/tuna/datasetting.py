from datasets import load_dataset
from textwrap import dedent
import pandas as pd
from .train import Model

import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import gc
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

def create_dataset(model, dataset):
    datasetter = Datasetter(model, dataset)
    del datasetter
    gc.collect()

class Datacreator():
    def __init__(self, dataset_name, context=False):
        self.filename = dataset_name + ".jsonl"
        exists = self.check_if_exists()
        if exists:
            num_fields = self.check_fields()
            if num_fields == 2 and context == True:
                print("Dataset already has only 2 fields and you are trying to input context. Exiting")
                exit(1)
            if num_fields == 3 and context == False:
                print("Dataset already has 3 fields and you are not inputting context. Exiting")
                exit(1)
            if num_fields < 2 or num_fields > 3:
                print("Invalid number of fields. Exitting")
                exit(1)
        with open(self.filename, "a") as file:
            while True:
                user_input = input("Input: ")
                if user_input.lower() == "exit":
                    break

                if context:
                    user_context = input("Context: ")
                    if user_context.lower() == "exit":
                        break


                user_output = input("Output: ")
                if user_output.lower() == "exit":
                    break

                if context:
                    entry = {"input": user_input, "context": user_context, "output": user_output}
                else:
                    entry = {"input": user_input, "output": user_output}
                file.write(json.dumps(entry) + "\n")

        print(f"Data saved to {self.filename}")

    def check_fields(self):
        dataset = load_dataset('json', data_files=self.filename)
        print(dataset.column_names)
        return len(dataset.column_names['train'])

    def check_if_exists(self):
        file = Path(self.filename)
        if file.exists():
            return True
        else:
            return False


class Datasetter():
    def __init__(self, model, dataset):
        self.dataset = load_dataset(dataset)
        self.model = Model(model)
        self.df = self.create_dataframe()
        # TODO Add this check
        # self.df.isnull().value_counts()
        self.df['text'] = self.df.apply(self.format_example, axis=1)

        self.df["token_count"] = self.df.apply(self.count_tokens, axis=1)
        print(self.df)
        print(self.df.text.iloc[0])

        # show_token_counts(df)

        # Remove samples larger than 512
        self.df = self.df[self.df.token_count < 512]

        # Only use 6000 of the samples
        self.df = self.df.sample(6000)
        self.df.shape

        train, temp = train_test_split(self.df, test_size=0.2)
        val, test = train_test_split(temp, test_size=0.2)

        percentage_train, percentage_val, percentage_test = len(train) / len(self.df), len(val) / len(self.df), len(test) / len(self.df)

        samples_train, samples_val, samples_test = len(train), len(val), len(test)

        print(f"Percentage of training data: {percentage_train}")
        print(f"Percentage of validation data: {percentage_val}")
        print(f"Percentage of test data: {percentage_test}")

        print(f"Number of training samples: {samples_train}")
        print(f"Number of validation samples: {samples_val}")
        print(f"Number of test samples: {samples_test}")

        train.sample(n=4000).to_json("train.jsonl", orient="records", lines=True)
        val.sample(n=500).to_json("val.jsonl", orient="records", lines=True)
        test.sample(n=100).to_json("test.jsonl", orient="records", lines=True)


    def create_dataframe(self):
        rows = []
        for item in self.dataset['train']:
            rows.append({
                "question": item["question"],
                "context": item["context"],
                "answer": item["answer"],
            })
        return pd.DataFrame(rows)



    def format_example(self, row: dict):
        prompt = dedent(
            f"""
        {row["question"]}

        Information:

        ```
        {row["context"]}
        ```
        """
        )
        messages = [
                {
                "role": "system",
                "content": "Use only the information to answer the question",
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": row["answer"]},
        ]
        return self.model.tokenizer.apply_chat_template(messages, tokenize=False)


    def count_tokens(self, row: dict) -> int:
        return len(
            self.model.tokenizer(
                row["text"],
                add_special_tokens=True,
                return_attention_mask=False
            )["input_ids"]
        )





    def show_token_counts(self):
        plt.hist(self.df.token_count, weights=np.ones(len(self.df.token_count)) / len(self.df.token_count))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel("Tokens")
        plt.ylabel("Percentage")
        plt.show()








