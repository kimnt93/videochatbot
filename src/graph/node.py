import base64
import logging

from langchain_core.runnables import RunnableConfig

from src.config import MAX_CONV_HISTORY
from src.factory.chain_factory import LlmChainFactory
from src.graph.state import ConversationState, IsMultiModalInput
from src.indexer.retriever import find_similar_from_semantic_text, find_similar_from_mmimg, find_similar_from_mmtext, \
    find_similar_from_kw_text
from src.utils import read_image_to_binary


def is_multimodal_input(state: ConversationState):
    logging.debug(f"Start chat: {state}")

    im_path = state.get("img_path", None)
    return IsMultiModalInput.YES if im_path is not None else IsMultiModalInput.NO


def summarize_conversation(state: ConversationState):
    """
    Summarize the conversation from the chat history
    :param state:
    :return:
    """
    logging.debug(f"Start summary: {state}")

    chat_history = state.get("chat_history", None)
    chat_summary = ""
    if chat_history is not None:
        if chat_history:
            conversation = "\n".join(chat_history)
            chain = LlmChainFactory.create_conversation_summary_chain()
            chat_summary = chain.invoke({"conversation": conversation})

    return {"chat_summary": chat_summary}


def transform_question(state: ConversationState):
    question = state['question']
    chain = LlmChainFactory.create_question_rewrite_chain()
    transformed_question = chain.invoke({"question": question})
    # transformed_question = question
    return {"transformed_question": transformed_question}


def grade_document(state: ConversationState):
    logging.debug(f"Start grade docs: {state}")

    question = state['question']
    documents = state['documents']
    graded_documents = []

    for doc in documents:
        chain = LlmChainFactory.create_grade_document_chain(document=doc.page_content)
        grade = chain.invoke({"question": question})
        if 'yes' in grade:
            graded_documents.append(doc.page_content)

    return {"documents": graded_documents}


def route_question(state: ConversationState):
    logging.debug(f"Start routing: {state}")
    question = state['question']
    chain = LlmChainFactory.create_question_routing_chain()
    response = chain.invoke({"question": question})
    return {"response": response}


def retrieve_documents(state: ConversationState):
    """
    Retrieve documents from database (semantic and text)
    :param state:
    :return:
    """
    logging.debug(f"Start retrieve: {state}")
    question = state['transformed_question']  # find using transformed question
    s_documents = find_similar_from_semantic_text(question, top_k=3)
    t_documents = find_similar_from_kw_text(question, top_k=3)
    return {"documents": s_documents + t_documents}


def retrieve_mm_documents(state: ConversationState):
    """
    Retrieve documents from database (semantic and text)
    :param state:
    :return:
    """
    logging.debug(f"Start mm retrieve: {state}")
    img_path = state['img_path']
    question = state['question']
    image_retrieved_documents = find_similar_from_mmimg(img_path, top_k=3)
    text_retrieved_documents = find_similar_from_semantic_text(question, top_k=3)
    documents = [doc.page_content for doc in text_retrieved_documents]
    documents.extend([doc.metadata['context'] for doc in image_retrieved_documents])
    return {"documents": documents}


def generate_response(state: ConversationState, config: RunnableConfig):
    logging.debug(f"Start response: {state}")
    question = state['question']
    documents = state['documents']

    chain = LlmChainFactory.create_rag_generate_chain(documents, state['chat_summary'])
    response = chain.invoke({"question": question}, config)

    # add history
    chat_history = state.get("chat_history", None)
    if chat_history is None:
        chat_history = []

    chat_history.insert(0, f"Human: {question}\nAI: {response}")
    chat_history = chat_history[:MAX_CONV_HISTORY + 1]  # keep only the last 5 chat history
    return {"chat_history": chat_history, "response": response}


def generate_mm_response(state: ConversationState, config: RunnableConfig):
    logging.debug(f"Start mm response: {state}")
    documents = state['documents']
    question = state['question']

    chain = LlmChainFactory.create_rag_multimodal_chain(documents, question, state['chat_summary'])
    image_data = read_image_to_binary(state['img_path'])
    image_datab64 = base64.b64encode(image_data).decode("utf-8")

    response = chain.invoke({"image_data": image_datab64}, config)

    # add history
    chat_history = state.get("chat_history", None)
    if chat_history is None:
        chat_history = []

    chat_history.insert(0, f"Human: {question}\nAI: {response}")
    chat_history = chat_history[:MAX_CONV_HISTORY + 1]  # keep only the last 5 chat history
    return {"chat_history": chat_history, "response": response}
