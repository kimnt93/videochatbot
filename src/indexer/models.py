from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
import datetime
from sqlalchemy.orm import relationship, mapped_column
from pgvector.sqlalchemy import Vector

from src.indexer.config import EMB_DIM

Base = declarative_base()


class UploadedVideo(Base):
    __tablename__ = "video_uploaded"
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(255), unique=True)  # task_id
    status = Column(String(255))
    video_name = Column(String(255))
    video_url = Column(String(255))
    checksum = Column(String(255), unique=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)
    is_disabled = Column(Boolean, default=False)


class VideoDetail(Base):
    __tablename__ = "video_detail"
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(255))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    # separate
    caption = Column(String(255))
    from_time = Column(String(255))
    to_time = Column(String(255))


# using lancedb for vector and image store
class VideoCaptionChunkEmbedding(Base):
    __tablename__ = "video_caption_chunk_embedding"
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(255))
    chunk_caption = Column(Text)
    chunk_id = Column(String(255))
    embedding = mapped_column(Vector(EMB_DIM))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class VideoFrameTextEmbedding(Base):
    __tablename__ = "video_frame_text_embedding"
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(255))
    frame_id = Column(String(255))
    text = Column(Text)
    frame = Column(Text)
    frame_path = Column(String(255))
    # store embeddings
    frame_embedding = mapped_column(Vector(EMB_DIM))
    text_embedding = mapped_column(Vector(EMB_DIM))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
