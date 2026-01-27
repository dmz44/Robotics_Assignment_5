#!/usr/bin/env python3
import os
from llama_cpp import Llama

# Additional example that extends test_llama_text with message history and system prompt.

MODEL_PATH = os.path.expanduser("~/llama.cpp/models/llama-2-7b-32k-instruct.Q4_K_M.gguf")
#MODEL_PATH = os.path.expanduser("~/llama.cpp/models/llama-2-7b-chat.Q4_K_M.gguf")

def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: model not found at {MODEL_PATH}")
        return

    llm = Llama(
        model_path=MODEL_PATH,

        # Modify this to add native support to various special formatting such as [INST] or <<SYS>>
        chat_format="llama-2", 

        n_ctx=512,
        n_threads=4,
        n_gpu_layers=33,
        seed=42,
        use_mlock=True
    )
    print("Model loaded. Type 'exit' to quit.")

    # Define your system prompt here
    system_message = {
        "role": "system",
        "content": "You are a helpful and concise assistant."
    }
    messages = [system_message]

    while True:
        prompt = input("\nYou: ")
        if prompt.strip().lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": prompt})

        print("Assistant: ", end="", flush=True)

        full_reply = ""
        for chunk in llm.create_chat_completion(
            messages=messages,
            max_tokens=128,
            stop=["\nYou:"],
            temperature=0.7,
            top_p=0.95,
            stream=True
        ):
            # Check if the chunk has content
            delta = chunk['choices'][0].get('delta', {})
            token = delta.get('content', '')

            if token:
                full_reply += token
                print(token, end="", flush=True)
        
        if full_reply:
            messages.append({"role": "assistant", "content": full_reply})
        
        print()

if __name__ == "__main__":
    main()

