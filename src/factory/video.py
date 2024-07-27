import logging
import re
from src.core.video.local import LocalVideoClient
from src.core.video.youtube import YoutubeVideoClient


def create_video_client(url):
    # Regex to check if the URL is a YouTube URL
    youtube_regex = re.compile(r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/')

    if youtube_regex.match(url):
        logging.info(f"Using YoutubeVideoClient for {url}")
        return YoutubeVideoClient(url)
    else:
        # check if url is a local file
        logging.info(f"Using LocalVideoClient for {url}")
        return LocalVideoClient(url)
