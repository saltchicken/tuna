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

