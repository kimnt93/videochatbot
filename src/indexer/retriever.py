import hashlib
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.indexer.config import PERSIST_VECDB_TEXT, PERSIST_VECDB_MMIMG, PERSIST_VECDB_MMTEXT
from src.indexer.mmeb import MM_IMG_EMBEDDING, MM_TEXT_EMBEDDING
from src.llm.model import EMBEDDING_MODEL

vectordb_text = Chroma(persist_directory=PERSIST_VECDB_TEXT, embedding_function=EMBEDDING_MODEL)
vectordb_mmtext = Chroma(persist_directory=PERSIST_VECDB_MMIMG, embedding_function=MM_TEXT_EMBEDDING)
vectordb_mmimg = Chroma(persist_directory=PERSIST_VECDB_MMTEXT, embedding_function=MM_IMG_EMBEDDING)


def add_new_document_text(new_documents: List[Document]):
    return vectordb_text.add_documents(
        new_documents,
        ids=[hashlib.md5(doc.page_content.encode()).hexdigest() for doc in new_documents]
    )


def add_new_document_text_mm(new_documents: List[Document]):
    return vectordb_mmtext.add_documents(
        new_documents,
        ids=[hashlib.md5(doc.metadata.__str__().encode()).hexdigest() for doc in new_documents]
    )


def add_new_document_img_mm(new_documents: List[Document]):
    return vectordb_mmimg.add_documents(
        new_documents,
        ids=[hashlib.md5(doc.metadata.__str__().encode()).hexdigest() for doc in new_documents]
    )


def find_similar_from_text(embedding: str, top_k: int = 5):
    pass


def find_similar_from_mmimg(img_path: str, top_k: int = 5):
    pass


def find_similar_from_mmtext(text: str, top_k: int = 5):
    pass
