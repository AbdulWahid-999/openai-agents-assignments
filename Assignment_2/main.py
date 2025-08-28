import asyncio
from decouple import config
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel, Agent, Runner , set_tracing_disabled

# Load Gemini API key from .env file
gemini_api_key = config("GEMINI_API_KEY")
set_tracing_disabled(True)  # Disable tracing for cleaner output

# Setup Gemini client using OpenAI-compatible endpoint
client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
)

# Configure the Gemini model
MODEL = OpenAIChatCompletionsModel(
    openai_client=client,
    model="gemini-2.5-flash",  # Or use "gemini-1.5-pro" if you want
)

# Define the FAQ Agent
faq_agent = Agent(
    name="FAQBot",
    instructions="""
You are a helpful FAQ bot. Only answer using the following predefined answers:

1. What is your name?
   - My name is FAQBot.

2. What can you do?
   - I can answer simple predefined questions to help users understand how to use this system.

3. How do I reset my password?
   - Go to the login page, click "Forgot Password", and follow the instructions.

4. What are your working hours?
   - I am available 24/7.

5. Who created you?
   - I was created using Google's Gemini model and a Python agent framework.

If the question is not in this list, reply:
"I'm not sure about that. Please ask something else."
""",
    model=MODEL,
)

# Run predefined questions
async def main():
    test_questions = [
        "What is your name?",
        "What can you do?",
        "How do I reset my password?",
        "Who created you?",
        "What are your working hours?",
        
    ]

    for question in test_questions:
        print("\nYou:", question)
        result = await Runner.run(faq_agent, question)
        print("FAQBot:", result.final_output)

# Execute the async main

asyncio.run(main())
