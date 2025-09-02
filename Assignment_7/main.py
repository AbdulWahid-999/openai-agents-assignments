from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled , function_tool
from dotenv import load_dotenv
from tavily import TavilyClient

import os


set_tracing_disabled(True)
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
base_url = os.getenv("GEMINI_BASE_URL")
model_name = os.getenv("GEMINI_MODEL_NAME")
tavily_api_key = os.getenv("TAVILY_API_KEY")


client = AsyncOpenAI(api_key=api_key, base_url=base_url)

model = OpenAIChatCompletionsModel(
    openai_client=client,
    model=model_name
)

tavily_client = TavilyClient(api_key=tavily_api_key)
set_tracing_disabled(True)  

@function_tool
def web_search(query: str) -> str:
    """Search the web using Tavily API and return summarized results."""
    results = tavily_client.search(query)

    if "results" in results and results["results"]:
        formatted = "\n".join([
            f"Title: {item.get('title', 'No title')}\n"
            f"URL: {item.get('url', '')}\n"
            f"Snippet: {item.get('content', '')}\n"
            for item in results["results"]
        ])
        return formatted

    return "No results found."




agent = Agent(
    name="Search Agent",
    instructions= "You are a helpful search agent. "
        "Always call the web_search tool if you need information. "
        "Use the returned snippets to form your answer instead of saying you cannot fetch.",
    model=model,
    tools = [web_search]
)

prompt = input("Enter your question : ")
result = Runner.run_sync(agent,prompt)
print(f"Agent: {result.final_output}")