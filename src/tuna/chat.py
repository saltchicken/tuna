import json
import requests
import ollama

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

def stream_ollama(model, prompt, system_message=None):
    ollama.api_host = "http://localhost:11434"
    messages = [{"role": "user", "content": prompt}]
    if system_message:
        messages.insert(0,{"role": "system", "content": system_message})

    stream = ollama.chat(
        model=model,  # Replace with the model you're using
        messages=messages,
        stream=True
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)


def query_ollama(model, prompt, system_message=None):
    ollama.api_host = "http://localhost:11434"
    messages = [{"role": "user", "content": prompt}]
    if system_message:
        messages.insert(0,{"role": "system", "content": system_message})

    response = ollama.chat(
        model=model,  # Replace with the model you're using
        messages=messages
    )

    print(response['message']['content'])
