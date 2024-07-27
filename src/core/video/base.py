import logging
import os
from abc import ABC, abstractmethod
import moviepy.editor as mp

from src import config
from src.core.video.transcriber import downsample_audio


class AClient(ABC):
    def __init__(self, url):
        self.url = url
        self.checksum = None
        self.download_url = None
        self.audio_url = None
        self.caption_url = None
        self.frame_dir = None
        self.video_title = None

    @abstractmethod
    def get_transcript(self):
        raise NotImplementedError()

    def video_to_frames(self, fps=1):
        self.frame_dir = os.path.join(config.DOWNLOAD_DIRECTORY, f"{self.checksum}_frames")
        if not os.path.exists(self.frame_dir):
            os.mkdir(self.frame_dir)
        clip = mp.VideoFileClip(self.download_url)
        clip.write_images_sequence(os.path.join(self.frame_dir, f"frame_%04d.png"), fps=fps)

    def convert_mp3(self):
        if os.path.exists(self.audio_url):
            logging.info("== Video is converted")
        else:
            clip = mp.VideoFileClip(self.download_url)
            clip.audio.write_audiofile(self.audio_url)
            # convert to 16000Hz
            # Whisper will downsample audio to 16,000 Hz mono before transcribing.
            # This preprocessing can be performed client-side to reduce file size and allow longer files to be uploaded to groq
            # Doc: https://console.groq.com/docs/speech-text
            downsample_audio(self.audio_url)

        return self

    @abstractmethod
    def download_video(self):
        raise NotImplementedError()
