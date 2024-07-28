import hashlib
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.indexer.config import PERSIST_VECDB_TEXT, PERSIST_VECDB_MMIMG, PERSIST_VECDB_MMTEXT
from src.indexer.mmeb import MM_IMG_EMBEDDING, MM_TEXT_EMBEDDING
from src.llm.model import EMBEDDING_MODEL
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.documents import Document


vectordb_text = Chroma(persist_directory=PERSIST_VECDB_TEXT, embedding_function=EMBEDDING_MODEL)
vectordb_mmtext = Chroma(persist_directory=PERSIST_VECDB_MMIMG, embedding_function=MM_TEXT_EMBEDDING)
vectordb_mmimg = Chroma(persist_directory=PERSIST_VECDB_MMTEXT, embedding_function=MM_IMG_EMBEDDING)

# in memory retriever
bm25_retriever = BM25Retriever.from_documents(
    [Document(page_content=page_content) for page_content in vectordb_text.get()['documents']],
)
bm25_retriever.k = 10

# convert retriever


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


# search retriever
def find_similar_from_semantic_text(text: str, top_k: int = 5):
    sen_docs = vectordb_text.similarity_search(text, k=top_k)
    return sen_docs


def find_similar_from_kw_text(text: str, top_k: int = 5):
    kw_docs = bm25_retriever.invoke(text)[:top_k]
    return kw_docs


def find_similar_from_mmimg(img_path: str, top_k: int = 5):
    return vectordb_mmimg.similarity_search(img_path, k=top_k)


def find_similar_from_mmtext(text: str, top_k: int = 5):
    return vectordb_mmtext.similarity_search(text, k=top_k)

