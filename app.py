import logging

import chainlit as cl
from langchain_core.runnables import Runnable, RunnableConfig
from langsmith import traceable

from src.factory.graph_factory import create_chatbot_default_workflow


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
    msg = cl.Message(content="")
    output_msg = ""

    chat_end_event = dict()
    async for event in graph.astream_events({"question": question, "img_path": images}, version="v2"):
        logging.info(event)
        # Update chat message with event response
        #
        # {'event': 'on_chat_model_stream', 'name': 'ChatGroq', 'run_id': '424611c6-ed2b-4567-a4bd-301c33a64f9e',
        # 'tags': ['seq:step:2'], 'metadata': {'langgraph_step': 5, 'langgraph_node': 'generate_response',
        # 'langgraph_triggers': ['grade_document'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'},
        # 'data': {'chunk': AIMessageChunk(content='', id='run-424611c6-ed2b-4567-a4bd-301c33a64f9e')}, 'parent_ids': []}
        if event['event'] == "on_chat_model_stream":
            content = event["data"]["chunk"].content or ""
            await msg.stream_token(token=content)
            output_msg += content
    #     if event['event'] == "on_chat_end":
    #         chat_end_event = event
    #
    # try:
    #     graph['chat_history'] = chat_end_event['data']['output']['chat_history']
    # except:
    #     pass

    await msg.send()
    cl.user_session.set("graph", graph)  # update graph state
    return output_msg
