from agents import Agent, Runner, OpenAIChatCompletionsModel, function_tool
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")


client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASE_URL
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client
)


@function_tool
def add(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b

@function_tool
def get_weather(city: str) -> str:
    """Get fake weather info for a city."""
    fake_weather = {
        "karachi": " Sunny, 32°C",
        "lahore": " Partly Cloudy, 30°C",
        "islamabad": " Thunderstorms, 28°C"
    }
    return fake_weather.get(city.lower(), f"Weather for {city} not found.")



agent = Agent(
    name="MultiToolAgent",
    model=model,
    instructions="You are a smart assistant. Use tools when needed (math or weather).",
    tools=[add, get_weather]
)

def main():
    test_questions = [
        "Can you add 12 and 8?",
        "What's the weather in Karachi?",
        "Add 55 and 45 please.",
        "Tell me weather in Islamabad.",
    ]

    for q in test_questions:
        print(f"\n Question: {q}")
        runner = Runner.run_sync(agent, q)
        print(f"Answer: {runner.final_output}")



main()
