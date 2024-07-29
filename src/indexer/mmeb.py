"""
Multimodal embedding for image, text and audio
https://github.com/facebookresearch/ImageBind
"""
from typing import List

from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind import data
import torch

from src.llm.model import EMBEDDING_MODEL

# MM_EMBEDDING = imagebind_model.imagebind_huge(pretrained=True)
# MM_EMBEDDING.eval()
# device = "cuda" if torch.cuda.is_available() else "cpu"


def get_image_embedding_mm(image_path):
    image_path = [image_path]
    inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_path, device)}
    with torch.no_grad():
        embedding = MM_EMBEDDING(inputs)

    vec = embedding['vision'].reshape(-1)
    vec = vec.numpy()
    return vec


def get_text_embedding_mm(text):
    texts = [text]
    inputs = {ModalityType.TEXT: data.load_and_transform_text(texts, device)}
    with torch.no_grad():
        embedding = MM_EMBEDDING(inputs)

    vec = embedding['text'].reshape(-1)
    vec = vec.numpy()
    return vec


def get_text_embedding_txt(text):
    return EMBEDDING_MODEL.embed_query(text)


def get_doc_embedding_txt(text):
    return EMBEDDING_MODEL.embed_documents(text)


class MultimodalImageEmbedding:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, im_paths: List[str]) -> List[List[float]]:
        return [get_image_embedding_mm(t).tolist() for t in im_paths]

    def embed_query(self, im_path: str) -> List[float]:
        return get_image_embedding_mm(im_path).tolist()


class MultimodalTextEmbedding:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [get_text_embedding_mm(t).tolist() for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return get_text_embedding_mm(text).tolist()


MM_IMG_EMBEDDING = MultimodalImageEmbedding(None)
MM_TEXT_EMBEDDING = MultimodalTextEmbedding(None)
