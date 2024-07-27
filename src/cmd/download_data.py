import logging
import os

from tqdm import tqdm

from src import config
from src.core.video.client import VideoClient

if not os.path.exists(config.DOWNLOAD_DIRECTORY):
    os.mkdir(config.DOWNLOAD_DIRECTORY)


def download_data():
    logging.info("Download videos")
    all_videos = [url.strip() for url in open("videos.txt").readlines()]

    # download all data
    logging.info("Preprocessing")
    for url in tqdm(all_videos):
        video_client = VideoClient(url)
        video_client.download_video()
        video_client.download_mp3()
        video_client.get_transcript()
        video_client.video_to_frames(fps=config.INDEX_VIDEO_FPS)


if __name__ == "__main__":
    download_data()
