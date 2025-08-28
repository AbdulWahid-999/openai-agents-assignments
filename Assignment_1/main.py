
from agents import Agent, Runner, OpenAIChatCompletionsModel , AsyncOpenAI , set_tracing_disabled
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file if present
set_tracing_disabled(True)
 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL") 

client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASE_URL)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client,   
)

agent = Agent(
    name="SimpleAgent",
    model=model,
    instructions="You are a helpful assistant , when someone ask about your name answer with I'm Batman  ."
)


prompt = input("Enter your prompt: ")
runner = Runner.run_sync(agent, prompt)
print("Response:", runner.final_output)

