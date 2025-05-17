import requests
import json

response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer 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",
        "Content-Type": "application/json",
    },
    data=json.dumps({
        "model": "meta-llama/llama-3.1-8b-instruct:free",
        "messages": [
            {
                "role": "user",
                "content": "What is the meaning of life?"
            }
        ],
    })
)

if response.ok:
    print("Response received:")
    print(response.json())  # Print the JSON response
else:
    print(f"Error: {response.status_code}")
    print(response.text)