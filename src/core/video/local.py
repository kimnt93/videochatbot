import logging
import os.path
import shutil
import json

from src import config
from src.core.video.base import AClient
from src.core.video.transcriber import transcript_audio, VideoTranscript
from src.core.video.utils import calculate_checksum


class LocalVideoClient(AClient):
    def __init__(self, url):
        super().__init__(url)
        self.checksum = calculate_checksum(url, 10000)
        self.download_url = os.path.join(config.DOWNLOAD_DIRECTORY, f"{self.checksum}{os.path.splitext(url)[-1]}")
        self.audio_url = os.path.join(config.DOWNLOAD_DIRECTORY, f"{self.checksum}.mp3")
        self.caption_url = os.path.join(config.DOWNLOAD_DIRECTORY, f"{self.checksum}.json")

    def get_transcript(self):
        if os.path.exists(self.caption_url):
            logging.info("== Video is transcript")
            # load transcript
            with open(self.caption_url, 'r') as f:
                jt = json.load(f)
                rt = VideoTranscript().from_dict(jt)
            return rt

        rt: VideoTranscript = transcript_audio(self.audio_url)
        return rt

    def download_video(self):
        logging.info(f"Copy the video file to the downloads directory: src: {self.url}, dest: {self.download_url}")
        shutil.copy(self.url, self.download_url)
        # ./resources/video demo.mp4 -> video demo
        self.video_title = os.path.splitext(os.path.basename(self.url))[0]
        return self
