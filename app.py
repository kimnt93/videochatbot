import logging

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
    cl.user_session.set("graph", graph)


@cl.on_message
@traceable
async def on_message(message: cl.Message):
    question = message.content
    if message.elements:
        images = [file for file in message.elements if "image" in file.mime]  # only support 1 image
        images = images[0].path

    else:
        images = None

    logging.info(f"Start chat `{question}` || {images}")
    graph: Runnable = cl.user_session.get("graph")
    state = graph.invoke({"question": question, "img_path": images})
    await cl.Message(state['response']).send()

    cl.user_session.set("graph", graph)
