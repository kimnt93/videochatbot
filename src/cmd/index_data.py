import glob
import logging
import os

from langchain_core.documents import Document
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.core.video.transcriber import VideoTranscript
from src.indexer.retriever import add_new_document_text, add_new_document_text_mm, add_new_document_img_mm

if not os.path.exists(config.DOWNLOAD_DIRECTORY):
    os.mkdir(config.DOWNLOAD_DIRECTORY)


def index_data():
    logging.info("Index data")

    logging.info("Process text data to vectorstore")
    # index vector, text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        separators=[
            "\n\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
        # Existing args
    )

    for file_path in glob.glob(os.path.join(config.DOWNLOAD_DIRECTORY, "*.json")):
        video_id = os.path.basename(file_path).replace(".json", "")
        video_meta = open(file_path, 'r').read()
        vit = VideoTranscript()
        vit = vit.from_dict(video_meta)

        logging.info("Indexing text...")
        # index vector
        # split chunk and add title
        # each chunk have format
        # <title>\n\n<chunk content>
        doc_chunks = text_splitter.create_documents([vit.vvt])
        docs = []
        for doc in doc_chunks:
            # index text: title + chunk content
            doc = Document(
                page_content=f"{vit.title}\n\n{doc.page_content}",
                metadata={"title": vit.title, "video_id": video_id}
            )
            docs.append(doc)
        add_new_document_text(docs)  # index text

        logging.info("Indexing multimodal")
        # now, embed multimodal
        # current frame timestamp [+/- 10 subtitle-lines] as text (total 20 seconds)
        # frame -> emb
        # text -> emb
        for frame_dir in tqdm(sorted(glob.glob(os.path.join(config.DOWNLOAD_DIRECTORY, f"{video_id}*_frames/*.png")))):
            base_name = os.path.split(os.path.normpath(frame_dir))[-1]
            frame_current_sec = float(base_name.split('_')[-1].replace(".png", "")) / config.INDEX_VIDEO_FPS
            # scan in the subtitle
            current_subs = list()
            for idx, segment in enumerate(vit.segments):
                if segment.start > frame_current_sec:
                    break
                current_subs = vit.segments[max(idx - config.FRAME_SUB_CONTEXT, 0):min(idx + config.FRAME_SUB_CONTEXT, len(vit.segments))]

            # build context of a frame
            frame_context = " ".join([sub.text for sub in current_subs])
            logging.debug(frame_context)
            img_doc = Document(
                page_content=frame_dir,
                metadata={"title": vit.title, "video_id": video_id, "context": frame_context}
            )

            text_doc = Document(
                page_content=frame_context,
                metadata={"title": vit.title, "video_id": video_id}
            )

            # add document
            add_new_document_text_mm([text_doc])
            add_new_document_img_mm([img_doc])


if __name__ == "__main__":
    index_data()
