import asyncio

from decouple import config
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel, Agent, Runner , set_tracing_disabled , function_tool


# Load Gemini API key from .env
GEMINI_API_KEY = config("GEMINI_API_KEY")
set_tracing_disabled(True)

# Use Gemini via OpenAI-compatible endpoint
client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
)

# Use Gemini model
MODEL = OpenAIChatCompletionsModel(
    openai_client=client,
    model="gemini-2.5-flash",
)

# Step 1: Define the math function
@function_tool
def add(a: float, b: float) -> float:
    """Add two numbers"""
    print("tool fire")
    return a + b

# Step 2: Register the tool


# Step 3: Create the agent
math_agent = Agent(
    name="MathToolAgent",
    instructions="""
You are a smart math agent. If someone asks a question involving addition, you should call the 'add' tool to calculate the result.
Do not try to calculate yourself. Use the tool when it's appropriate.
""",
    model=MODEL,
    tools=[add],  # Tool registered here
)

# Step 4: Test with 3 questions
async def main():
    test_questions = [
        "What is 5 + 7?",
        "Can you add 12 and 8 for me?",
        "What's the sum of 100 and 350?",
        "How are you today?"  # Should not use tool
    ]

    for question in test_questions:
        print("\nYou:", question)
        result = await Runner.run(math_agent, question)
        print("Agent:", result.final_output)

# Run it
asyncio.run(main())
