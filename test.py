import openai

openai_api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
    model="gpt-4-1106-preview",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message["content"])