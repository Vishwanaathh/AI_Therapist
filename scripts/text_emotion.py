import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-1.5-flash")

response = model.generate_content(
    "You are VishwAI, an empathetic AI therapist. User: I feel stressed."
)

print(response.text)
