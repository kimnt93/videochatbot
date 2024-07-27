import chainlit as cl
from langchain_core.runnables import Runnable, RunnableConfig
from langsmith import traceable

from src.factory.graph_factory import create_chatbot_default_workflow
from src.graph.state import ConversationState


@cl.on_chat_start
async def on_chat_start():
    # start graph
    graph = create_chatbot_default_workflow()

    # save graph and state to the user session
    cl.user_session.set("graph", graph.compile())


@cl.on_message
@traceable
async def on_message(message: cl.Message):
    question = message.content
    if message.elements:
        images = [file for file in message.elements if "image" in file.mime][0]  # only support 1 image
    else:
        images = None

    graph: Runnable = cl.user_session.get("graph")

    # Append the new message to the state
    msg = cl.Message(content="")
    output_msg = ""
    async for chunk in graph.astream(
        {"question": question, "im_path": images},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)
        output_msg += chunk

    await msg.send()
    return output_msg
