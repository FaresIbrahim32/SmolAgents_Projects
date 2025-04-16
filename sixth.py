from smolagents import CodeAgent,LiteLLMModel
from dotenv import load_dotenv

load_dotenv()

import os

model = LiteLLMModel(
    model_id="huggingface/meta-llama/Llama-3.3-70B-Instruct",  # or another Claude model version
    temperature=0.2,
)
messages = []

while True:
    user_input= input("Enter a prompt: ")
    
    messages.append({"role":"user","content":user_input})
    
    response = model(messages,max_tokens=500)
    assistant_message = response.content
    
    print("Assistant:",assistant_message)
    messages.append({"role":"assistant","content":assistant_message})
    
    if user_input == "exit":
        break