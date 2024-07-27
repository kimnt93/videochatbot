import os


DATABASE_URL = os.getenv("DATABASE_URL")
EMB_DIM = 1024  # dimension of multimodal embedding
PERSIST_VECDB_TEXT = "./db/text"
PERSIST_VECDB_MMIMG = "./db/mmimg"
PERSIST_VECDB_MMTEXT = "./db/mmtext"