from typing import TypedDict, List, Dict, Union
from dataclasses import dataclass


class ConversationState(TypedDict):
    # The Human question/input
    question: str
    transformed_question: str

    # The chat history
    # List of: ["Human: <....>\nAI: <....>"]
    chat_history: Union[List[str], str]

    # The chat summary, the video_id use for get chat summary later on Celery
    chat_summary: str
    chat_summary_task_id: str

    # the context for the RAG generation
    documents: List[str]
    img_documents: List[str]  # for multimodal
    # 2 retrieval methods
    s_documents: List[str]
    t_documents: List[str]

    # for image retrieval
    image_retrieved_documents: List[str]

    # Count the number of steps for each node
    # This is useful for limit the number of steps for each node.
    step_counter: Dict[str, int]

    # final user response
    response: str

    # Add multimodal
    img_path: str


@dataclass
class RouteQueryNextState:
    RETRIEVE = 1
    NO_RETRIEVE = 2


@dataclass
class IsMultiModalInput:
    YES = 1
    NO = 2
