"""
Using transcriptions and translations APIs
https://console.groq.com/docs/speech-text
"""
import datetime
import os
import logging
from dataclasses import dataclass
from typing import List, Union
from pydub import AudioSegment
import src.llm.model as llm
from langsmith import traceable
from src import config
from src.factory.chain_factory import LlmChainFactory


@dataclass
class Segment:
    start: float
    end: float
    text: str


class VideoTranscript:
    def __init__(self):
        self.segments: List[Segment] = []
        self.full_text = ""
        self.title = ""
        self.summary = ""
        self.vvt = ""

    def format(self):
        """Convert to vvt format"""
        if self.summary == "":
            logging.info("Summary video transcript...")
            chain = LlmChainFactory.create_transcript_summary_chain()
            self.summary = chain.invoke({"subtitle": self.full_text})

        self.vvt = f"Title: {self.title}\n\n"
        self.vvt += f"Summary: {self.summary}\n\n"
        for segment in self.segments:
            self.vvt += f"{datetime.timedelta(seconds=segment.start)} --> {datetime.timedelta(seconds=segment.end)}\n{segment.text.strip()}\n\n"
        return self

    def to_dict(self):
        self.format()
        return {
            "segments": [segment.__dict__ for segment in self.segments],
            "full_text": self.full_text,
            "summary": self.summary,
            "vvt": self.vvt
        }

    def from_dict(self, jt: Union[dict, str]):
        if isinstance(jt, str):
            jt = eval(jt)

        self.segments = [Segment(**segment) for segment in jt['segments']]
        self.full_text = jt['full_text']
        self.summary = jt['summary']
        self.vvt = jt['vvt']
        return self


def downsample_audio(audio_path):
    """
    Downsample audio to 16kHz
    :param audio_path:
    :return:
    """
    logging.info(f"Downsample audio {audio_path} to 16kHz...")
    audio = AudioSegment.from_file(audio_path)
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
        audio.export(audio_path, format="mp3")


def chunk_audio(audio_path):
    file_name = os.path.basename(audio_path)
    file_size = os.path.getsize(audio_path)
    audio_list = []

    # Get length of audio file
    audio = AudioSegment.from_mp3(audio_path)
    logging.info(f'== Audio duration: {audio.duration_seconds:.2f} seconds')

    if file_size > config.MAX_AUDIO_SIZE_IN_MB * 1024 * 1024:
        logging.info(f'== The audio file is too large: {(file_size / 1024 / 1024):.2f} MB (>{config.MAX_AUDIO_SIZE_IN_MB}MB), chunking...')

        # Check if chunks already exist
        chunk_base_path = f"downloads/whisper/{file_name.split('.')[0]}"
        if os.path.exists(f"{chunk_base_path}_0.mp3"):
            logging.info('== Chunks already exist, loading...')
            for i in range(100):
                chunk_name = f"{chunk_base_path}_{i}.mp3"
                if os.path.exists(chunk_name):
                    audio_list.append(chunk_name)
                else:
                    return audio_list

        # PyDub handles time in milliseconds
        chunk_length = config.MAX_AUDIO_SIZE_IN_MB * 60 * 1000

        # Split the audio file into chunks
        for i, chunk in enumerate(audio[::chunk_length]):
            chunk_name = f"downloads/chunks/{file_name.split('.')[0]}_{i}.mp3"
            if not os.path.exists(chunk_name):
                chunk.export(chunk_name, format="mp3")
            audio_list.append(chunk_name)
    else:
        audio_list.append(audio_path)

    return audio_list


@traceable
def transcript_audio(audio_path: str) -> VideoTranscript:
    """Translate / transcribe audio to text.

    Args:
        audio_path (str): The audio path to translate / transcribe.
    Returns:
        str: The translated text.
    """
    audio_list = chunk_audio(audio_path)

    transcriptions = []
    segments = []
    durations = []

    for audio_file in audio_list:
        logging.info(f'== Transcribing {audio_file}...')
        with open(audio_file, "rb") as file:
            response = llm.GROQ_CLIENT.audio.transcriptions.create(
                model="whisper-large-v3",
                file=file,
                response_format="verbose_json"
            )
        if "error" in response:
            raise Exception(f"Transcription error: {response['error']['message']}")

        transcriptions.append(response.text.strip())
        segments.append(response.model_extra['segments'])
        audio = AudioSegment.from_mp3(audio_file)
        durations.append(audio.duration_seconds)

    # Merge multiple video segments
    rt = VideoTranscript()
    rt.full_text = ' '.join(transcriptions)
    logging.info(f'== Total words: {len(rt.full_text.split())} -- characters: {len(rt.full_text)}')

    total_durations = 0
    for audio_segments, duration in zip(segments, durations):
        total_durations += duration
        for segment in audio_segments:
            current_start = segment['start']
            current_end = segment['end']
            current_caption = segment['text']
            if rt.segments and current_start < rt.segments[-1].start and current_end < rt.segments[-1].end:
                current_start = total_durations + current_start
                current_end = total_durations + current_end
            rt.segments.append(Segment(start=current_start, end=current_end, text=current_caption))

    return rt
