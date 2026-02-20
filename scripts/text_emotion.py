import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("models/gemini-2.5-flash")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    prompt = f"""
    You are VishwAI, a calm and empathetic AI therapist.
    Be supportive and conversational.

    User: {user_input}
    """

    response = model.generate_content(prompt)

    print("VishwAI:", response.text)