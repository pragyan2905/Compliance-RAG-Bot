import requests
import os
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.environ.get("GROQ_API_KEY")
res = requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {api_key}"})
models = res.json().get("data", [])
for m in models:
    if "llama" in m["id"].lower():
        print(m["id"])
