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
    def create_question_routing_chain():
        return (
                ChatPromptTemplate.from_template(template=prompt.QUERY_ROUTING_PROMPT)
                | llm.LLAMA_8B_LLM
                | StrOutputParser()
        )

    @staticmethod
    def create_grade_document_chain(document: str):
        return (
                ChatPromptTemplate.from_template(
                    template=prompt.GRADE_DOCUMENT_PROMPT,
                    partial_variables={
                        "document": document
                    }
                )
                | llm.LLAMA_8B_LLM
                | StrOutputParser()
        )

    @staticmethod
    def create_rag_generate_chain(documents: Union[List[str], None], chat_summary: str):
        if documents is None:
            documents = []

        return (
                ChatPromptTemplate.from_template(
                    template=prompt.RAG_GENERATION_PROMPT,
                    partial_variables={
                        "today": datetime.datetime.now(),
                        "context": "\n----\n".join(documents),
                        "chat_summary": chat_summary
                    }
                )
                | llm.LLAMA_8B_LLM
                | StrOutputParser()
        )

    @staticmethod
    def create_rag_multimodal_chain(documents: List[str], question: str, chat_summary: str):
        """
        https://python.langchain.com/v0.2/docs/how_to/multimodal_prompts/
        :return:
        """
        sys_prompt = prompt.MULTIMODAL_DOCUMENT_PROMPT.format(question=question, today=datetime.datetime.now(), context="\n----\n".join(documents), chat_summary=chat_summary)
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "user",
                    [
                        {
                            "type": "text",
                            "text": sys_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
                        }
                    ],
                ),
            ]
        )

        return (
            chat_prompt
            | llm.MULTIMODAL_LLM
            | StrOutputParser()
        )
