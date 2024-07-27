from src.config import MAX_CONV_HISTORY
from src.factory.chain_factory import LlmChainFactory
from src.graph.state import ConversationState, IsMultiModalInput


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
    summary = ""
    if chat_history is not None:
        if chat_history:
            conversation = "\n".join(chat_history)
            chain = LlmChainFactory.create_conversation_summary_chain()
            summary = chain.invoke({"conversation": conversation})

    return {"chat_summary": summary}


def transform_question(state: ConversationState):
    question = state['question']
    chain = LlmChainFactory.create_question_rewrite_chain()
    transformed_question = chain.invoke({"question": question})
    return {"transformed_question": transformed_question}


def grade_document(state: ConversationState):
    question = state['question']

    semantic_retrieved_documents = state['semantic_retrieved_documents']
    text_retrieved_documents = state['text_retrieved_documents']

    for sdoc, tdoc in zip(semantic_retrieved_documents, text_retrieved_documents):
        chain = LlmChainFactory.grade_document_chain(documents=[{"d1": sdoc, "d2": tdoc}])
        grade = chain.invoke({"question": question})
        print(grade)


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
    question = state['question']
    semantic_retrieved_documents = list()
    text_retrieved_documents = list()
    return {"vector_retrieved_documents": semantic_retrieved_documents, "text_retrieved_documents": text_retrieved_documents}


def retrieve_mm_documents(state: ConversationState):
    """
    Retrieve documents from database (semantic and text)
    :param state:
    :return:
    """
    question = state['question']
    image_retrieved_documents = list()
    return {"image_retrieved_documents": image_retrieved_documents}


def generate_response(state: ConversationState):
    question = state['question']
    documents = state['documents']
    chain = LlmChainFactory.create_rag_generate_chain(documents)
    response = chain.invoke({"question": question})

    # add history
    chat_history = state.get("chat_history", None)
    if chat_history is None:
        chat_history = []

    chat_history.insert(0, f"Human: {question}\nAI: {response}")
    chat_history = chat_history[:MAX_CONV_HISTORY + 1]  # keep only the last 5 chat history
    return {"chat_history": chat_history, "response": response}

def generate_mm_response(state: ConversationState):
    pass

