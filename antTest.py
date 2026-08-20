import os

import anthropic
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

# Initialize the client with your API key
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Call the create method
message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Say hello and tell me a fun fact about Python."}
    ],
)

# Print the response
print(message.content[0].text)
