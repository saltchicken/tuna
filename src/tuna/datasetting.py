from datasets import load_dataset
from textwrap import dedent
import pandas as pd
from __main__ import Model

dataset = load_dataset("virattt/financial-qa-10k")
model = Model("unsloth/llama-3-8b-Instruct-bnb-4bit")

def create_dataframe(dataset):
    rows = []
    for item in dataset['train']:
        rows.append({
            "question": item["question"],
            "context": item["context"],
            "answer": item["answer"],
        })
    return pd.DataFrame(rows)

df = create_dataframe(dataset)

# TODO Add this check
df.isnull().value_counts()


def format_example(row: dict):
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
    return model.tokenizer.apply_chat_template(messages, tokenize=False)

df['text'] = df.apply(format_example, axis=1)

