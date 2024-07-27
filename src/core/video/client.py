import json
import os

from src import config
from src.factory.video import create_video_client
import moviepy.editor as mp


class VideoClient:
    def __init__(self, url):
        self.client = create_video_client(url)

    def video_to_frames(self, fps=1):
        return self.client.video_to_frames(fps)

    def get_transcript(self):
        rt = self.client.get_transcript()
        with open(self.client.caption_url, 'w') as f:
            f.write(json.dumps(rt.to_dict(), indent=2))

        return self

    def download_mp3(self):
        return self.client.convert_mp3()

    def download_video(self):
        return self.client.download_video()
