from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, END, START
import src.graph.node as node
from src.graph.state import ConversationState, RouteQueryNextState, IsMultiModalInput


def create_chatbot_default_workflow() -> Runnable:
    workflow = StateGraph(ConversationState)
    workflow.add_node("summarize_conversation", node.summarize_conversation)
    # workflow.add_node("route_question", node.route_question)
    workflow.add_node("transform_question", node.transform_question)
    workflow.add_node("retrieve_documents", node.retrieve_documents)
    workflow.add_node("grade_document", node.grade_document)
    workflow.add_node("generate_response", node.generate_response)

    # add multimodal nodes
    workflow.add_node("summarize_mm_conversation", node.summarize_conversation)
    workflow.add_node("retrieve_mm_documents", node.retrieve_mm_documents)
    workflow.add_node("generate_mm_response", node.generate_mm_response)

    # start and check if multimodal input
    workflow.add_conditional_edges(
        START,
        node.is_multimodal_input,
        {
            IsMultiModalInput.YES: "summarize_mm_conversation",  # multimodal input, summarize multimodal conversation
            IsMultiModalInput.NO: "summarize_conversation",      # no multimodal input, summarize conversation
        }
    )
    workflow.add_edge("summarize_mm_conversation", "retrieve_mm_documents")
    workflow.add_edge("retrieve_mm_documents", "generate_mm_response")
    workflow.add_edge("generate_mm_response", END)

    # not need to route
    workflow.add_edge("summarize_conversation", "transform_question")
    workflow.add_edge("transform_question", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "grade_document")
    workflow.add_edge("grade_document", "generate_response")
    workflow.add_edge("generate_response", END)

    app = workflow.compile()
    return app
