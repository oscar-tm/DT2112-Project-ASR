from tqdm import tqdm
import pandas as pd
import gc
import time

import torch
import whisperx
from faster_whisper import WhisperModel
from nemo.collections.asr.models import EncDecMultiTaskModel
from pocketsphinx import AudioFile

import logging
from nemo.utils import logging as nemo_logging

# Suppress a bunch of annoying warning/info messages
nemo_logging.setLevel(logging.WARNING)
logging.getLogger("whisperx").setLevel(logging.WARNING)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("nemo").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING, force=True)


class WhisperXHelper:
    def __init__(self):
        pass

    def init_model(self, model_size="large-v3"):
        # This will produce some warning but works
        self.model = whisperx.load_model(
            model_size,
            device="cuda",
            compute_type="float16",
            language="en",
        )

    def transcribe(self, path, batch_size=16):
        audio = whisperx.load_audio(path)
        segments = self.model.transcribe(audio, batch_size=batch_size)
        return "".join([segment["text"] for segment in segments["segments"]])

    def __str__(self):
        return "WisperX"

    def del_model(self):
        # Delete model in order to use next model on GPU
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


class FWhisperHelper:
    def __init__(self):
        pass

    def init_model(self, model_size="large-v3"):
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")

    def transcribe(self, path):
        segments, _ = self.model.transcribe(path, beam_size=5, language="en")
        return "".join([segment.text for segment in segments])

    def __str__(self):
        return "Faster Whisper"

    def del_model(self):
        # Delete model in order to use next model on GPU
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


class CanaryHelper:
    def __init__(self):
        pass

    def init_model(self):
        self.model = EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b")
        decode_cfg = self.model.cfg.decoding
        decode_cfg.beam.beam_size = 5
        self.model.change_decoding_strategy(decode_cfg)

    def transcribe(self, path):
        return self.model.transcribe(
            audio=[path],
            verbose=False,
            batch_size=16,  # batch size to run the inference with
        )[0]

    def __str__(self):
        return "Canary"

    def del_model(self):
        # Delete model in order to use next model on GPU
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


class SphinxHelper:
    def __init__(self):
        pass

    def init_model(self):
        # No explicit model initialization needed for CMU Sphinx
        pass

    def transcribe(self, path):
        audio = AudioFile(audio_file=path)
        return "".join([str(phrase) for phrase in audio])

    def __str__(self):
        return "CMU Sphinx"

    def del_model(self):
        # No need to delete anything as Sphinx doesn't use GPU memory
        pass


df = pd.read_csv("processed_dataset.csv", sep=";")
file_paths = df["path"].to_list()

models = [WhisperXHelper(), FWhisperHelper(), CanaryHelper(), SphinxHelper()]

for model in models:
    model.init_model()

    transcripts = [None] * len(file_paths)
    latency = [None] * len(file_paths)
    for i, path in tqdm(enumerate(file_paths), total=len(file_paths)):
        if i > 10:
            break
        start_time = time.perf_counter()

        transcripts[i] = model.transcribe(path)

        t_total = time.perf_counter() - start_time
        latency[i] = t_total

    df[str(model)] = transcripts
    df[str(model) + " Latency"] = latency

    model.del_model()

df.to_csv("STT_output.csv", sep=";", index=False)
