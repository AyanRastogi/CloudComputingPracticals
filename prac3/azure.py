import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure OpenAI settings
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  # Your Azure OpenAI endpoint
openai.api_version = "2023-05-15"  # Or the appropriate API version
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")    # Your Azure OpenAI API key

# Define a function to call the OpenAI API
def get_openai_response(prompt, deployment_name="your-deployment-name"):  # Replace with your deployment name
    try:
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

# Main program loop
if __name__ == "__main__":
    while True:
        user_prompt = input("Enter your prompt (or type 'exit'): ")
        if user_prompt.lower() == 'exit':
            break

        ai_response = get_openai_response(user_prompt)
        print("AI Response:", ai_response)