#!/usr/bin/env python3
import os
from llama_cpp import Llama

# Simple Llama example.

MODEL_PATH = os.path.expanduser("~/llama.cpp/models/llama-2-7b-32k-instruct.Q4_K_M.gguf")
#MODEL_PATH = os.path.expanduser("~/llama.cpp/models/llama-2-7b-chat.Q4_K_M.gguf")

def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: model not found at {MODEL_PATH}")
        return

    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=512,
        n_threads=4,
        n_gpu_layers=33,
        seed=42,
        use_mlock=True
    )
    print("Model loaded. Type 'exit' to quit.")

    while True:
        prompt = input("\nYou: ")
        if prompt.strip().lower() in ("exit", "quit"):
            break

        print("Assistant: ", end="", flush=True)
        full_reply = "" # Leaving it unused in this example.
        for chunk in llm(
            prompt=prompt,
            max_tokens=128,
            stop=["\nYou:"],
            echo=False,
            temperature=0.7,
            top_p=0.95,
            stream=True
        ):
            token = chunk["choices"][0]["text"]
            full_reply += token
            print(token, end="", flush=True)
        print()  

if __name__ == "__main__":
    main()

