import base64

from langchain_core.runnables import RunnableConfig
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from src.capp import run_chain_task, get_task_result
from src.config import MAX_CONV_HISTORY, MAX_GRADE_WORKER
from src.factory.chain_factory import LlmChainFactory
from src.graph.state import ConversationState, IsMultiModalInput
from src.indexer.retriever import find_similar_from_semantic_text, find_similar_from_mmimg, find_similar_from_mmtext, \
    find_similar_from_kw_text
from src.utils import read_image_to_binary


def is_multimodal_input(state: ConversationState):
    im_path = state.get("img_path", None)
    return IsMultiModalInput.YES if im_path is not None else IsMultiModalInput.NO


def summarize_conversation(state: ConversationState):
    """
    Summarize the conversation from the chat history
    :param state:
    :return:
    """
    chat_history = state.get("chat_history", None)
    task_id = ""
    if chat_history is not None:
        if chat_history:
            conversation = "\n".join(chat_history)
            chain = LlmChainFactory.create_conversation_summary_chain()
            data_input = {"conversation": conversation}
            task = run_chain_task.delay(chain, data_input)
            task_id = task.id

    return {"chat_summary": "", "chat_summary_task_id": task_id}


def transform_question(state: ConversationState):
    question = state['question']
    chain = LlmChainFactory.create_question_rewrite_chain()
    transformed_question = chain.invoke({"question": question})
    return {"transformed_question": transformed_question}


def grade_document(state: ConversationState):
    question = state['question']
    documents = state['documents']
    graded_documents = []

    def process_document(doc):
        chain = LlmChainFactory.create_grade_document_chain(document=doc.page_content)
        grade = chain.invoke({"question": question})
        return doc.page_content if 'yes' in grade else None

    # Limit the number of threads to a maximum of 4
    with ThreadPoolExecutor(max_workers=MAX_GRADE_WORKER) as executor:
        # Submit tasks to the thread pool
        futures = [executor.submit(process_document, doc) for doc in documents]

        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                graded_documents.append(result)

    return {"documents": graded_documents}


def route_question(state: ConversationState):
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
    img_path = state['img_path']
    question = state['question']
    image_retrieved_documents = find_similar_from_mmimg(img_path, top_k=3)
    text_retrieved_documents = find_similar_from_semantic_text(question, top_k=3)
    documents = [doc.page_content for doc in text_retrieved_documents]
    documents.extend([doc.metadata['context'] for doc in image_retrieved_documents])
    return {"documents": documents}


async def generate_response(state: ConversationState, config: RunnableConfig):
    question = state['question']
    documents = state['documents']
    state['chat_summary'] = get_task_result(state['chat_summary_task_id'])

    chain = LlmChainFactory.create_rag_generate_chain(documents, state['chat_summary'])
    response = await chain.ainvoke({"question": question}, config)

    # add history
    chat_history = state.get("chat_history", None)
    if chat_history is None:
        chat_history = []

    chat_history.insert(0, f"Human: {question}\nAI: {response}")
    chat_history = chat_history[:MAX_CONV_HISTORY + 1]  # keep only the last 5 chat history
    return {"chat_history": chat_history, "response": response}


async def generate_mm_response(state: ConversationState, config: RunnableConfig):
    documents = state['documents']
    question = state['question']
    state['chat_summary'] = get_task_result(state['chat_summary_task_id'])

    chain = LlmChainFactory.create_rag_multimodal_chain(documents, question, state['chat_summary'])
    image_data = read_image_to_binary(state['img_path'])
    image_datab64 = base64.b64encode(image_data).decode("utf-8")

    response = await chain.ainvoke({"image_data": image_datab64}, config)

    # add history
    chat_history = state.get("chat_history", None)
    if chat_history is None:
        chat_history = []

    chat_history.insert(0, f"Human: {question}\nAI: {response}")
    chat_history = chat_history[:MAX_CONV_HISTORY + 1]  # keep only the last 5 chat history
    return {"chat_history": chat_history, "response": response}
