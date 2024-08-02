import glob
import logging
import os
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config
from src.core.video.transcriber import VideoTranscript
from src.indexer.retriever import add_new_document_text

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
            page_content = doc.page_content

            # try:
            #     # index text: title + chunk content
            #     # merge subtitles to single sub
            #     pattern = re.compile(r'(\d+:\d+:\d+\.\d+ --> \d+:\d+:\d+\.\d+)\n(.*?)\n', re.DOTALL)
            #     matches = pattern.findall(doc.page_content)
            #     start_time = matches[0][0].split(' --> ')[0]
            #     end_time = matches[-1][0].split(' --> ')[1]
            #     texts = [match[1].replace('\n', ' ') for match in matches]
            #     combined_texts = ' '.join(texts)
            #     page_content = f"{start_time} --> {end_time}\n{combined_texts}"
            # except Exception as ex:
            #     pass

            page_content = page_content.strip()
            if page_content == "":
                continue
            doc = Document(
                page_content=page_content,
                metadata={"title": vit.title, "video_id": video_id}
            )
            docs.append(doc)
        add_new_document_text(docs)  # index text


if __name__ == "__main__":
    index_data()
