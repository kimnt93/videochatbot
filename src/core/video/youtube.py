import dataclasses
import datetime
import hashlib
import json
import logging
import os
from typing import List, Dict

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from src import config
from src.core.video.base import AClient
from src.core.video.transcriber import transcript_audio, VideoTranscript, Segment
from src.core.video.utils import extract_video_id


class YoutubeVideoClient(AClient):
    def __init__(self, url):
        super().__init__(url)
        self.video_id = extract_video_id(url)
        self.video_url = f"https://www.youtube.com/watch?v={self.video_id}" if self.video_id is not None else url
        self.checksum = hashlib.md5(self.video_id.encode()).hexdigest()

        self.download_url = os.path.join(config.DOWNLOAD_DIRECTORY, f"{self.checksum}.mp4")
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

        # continue
        default_language = None
        try:
            transcripts = YouTubeTranscriptApi.list_transcripts(self.video_id)

            # find transcript from default (auto-generated)
            for transcript in transcripts:
                """
                transcript.video_id,
                transcript.language,
                transcript.language_code,
                transcript.is_generated,
                transcript.is_translatable,
                transcript.translation_languages,
                """
                if transcript.language_code == 'en' and not transcript.is_generated:
                    default_language = transcript.language_code
                    break
                elif transcript.language_code in config.LANGUAGE_PRIORITIES:  # manual script
                    default_language = transcript.language_code
                    break

        except TranscriptsDisabled:
            logging.exception("TranscriptsDisabled error...")
        except Exception as ex:
            logging.exception(ex)
            # raise NotImplementedError("Unknown exception, download video and generate transcript using openai")

        if default_language is not None:
            """{'duration': 6.047, 'start': 6.353, 'text': 'cycle! Sadly, they’re kind of right to. But\xa0\nBeijing’s watching for more reasons than humor.\xa0'}"""
            subtitles = YouTubeTranscriptApi.get_transcript(self.video_id, languages=[default_language])
            rt = VideoTranscript()
            for s in subtitles:
                start = s['start']
                end = start + s['duration']
                text = s['text']
                rt.segments.append(Segment(start=start, end=end, text=text))

            # convert
            rt.full_text = " ".join([x['text'] for x in subtitles])
        else:
            pass
            # call whisper
            rt = transcript_audio(self.audio_url)
        #
        # if correct_transcript:
        #     transcript_content = generate_corrected_transcript(transcript_content, **correct_transcript_kwargs)

        return rt

    def download_video(self):
        if os.path.exists(self.download_url):
            logging.info("== Video is downloaded")
        else:
            yt_opts = {
                'verbose': True,
                'format': 'best[ext=mp4]',
                'outtmpl': self.download_url
            }

            with yt_dlp.YoutubeDL(yt_opts) as ydl:
                rt = ydl.extract_info(self.video_url, download=True)

            # add title
            self.video_title = rt.get('title', 'unknown')
            logging.info(f"Download video from url ```{self.video_url}```")
