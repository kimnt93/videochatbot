import datetime
from typing import List, Union, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import src.llm.model as llm
import src.llm.prompt as prompt


# prompt factory
class LlmChainFactory:

    @staticmethod
    def create_transcript_summary_chain():
        return (
                ChatPromptTemplate.from_template(
                    template=prompt.SUMMARY_TRANSCRIPT_PROMPT,
                    partial_variables={
                        "today": datetime.datetime.now()
                    }
                )
                | llm.LLAMA_8B_LLM
                | StrOutputParser()
        )

    @staticmethod
    def create_conversation_summary_chain():
        return (
                ChatPromptTemplate.from_template(
                    template=prompt.SUMMARY_CONVERSATION_PROMPT,
                    partial_variables={
                        "today": datetime.datetime.now()
                    }
                )
                | llm.LLAMA_8B_LLM
                | StrOutputParser()
        )

    @staticmethod
    def create_question_rewrite_chain():
        return (
                ChatPromptTemplate.from_template(
                    template=prompt.QUESTION_RE_WRITER_PROMPT,
                    partial_variables={
                        "today": datetime.datetime.now()
                    }
                )
                | llm.LLAMA_8B_LLM
                | StrOutputParser()
        )

    @staticmethod
    def create_rag_generate_chain(documents: Union[List[str], None]):
        if documents is None:
            documents = []

        return (
                ChatPromptTemplate.from_template(
                    template=prompt.RAG_GENERATION_PROMPT,
                    partial_variables={
                        "today": datetime.datetime.now(),
                        "context": documents
                    }
                )
                | llm.LLAMA_70B_LLM
                | StrOutputParser()
        )

    @staticmethod
    def create_question_routing_chain():
        return (
                ChatPromptTemplate.from_template(template=prompt.QUERY_ROUTING_PROMPT)
                | llm.LLAMA_8B_LLM
                | StrOutputParser()
        )

    @staticmethod
    def grade_document_chain(documents: List[Dict[str, str]]):
        if documents is None:
            documents = []

        return (
                ChatPromptTemplate.from_template(
                    template=prompt.GRADE_DOCUMENT_PROMPT,
                    partial_variables={
                        "today": datetime.datetime.now(),
                        "documents": [{"d" + str(i + 1): doc} for i, doc in enumerate(documents)]
                        # with prompt example format
                    }
                )
                | llm.LLAMA_8B_LLM
                | StrOutputParser()
        )
