import chainlit as cl
from main import agent
from agents import Runner

@cl.on_message
async def main(message: cl.Message):
    result = await Runner.run(agent , message.content)
    await cl.Message(content=result.final_output).send()
