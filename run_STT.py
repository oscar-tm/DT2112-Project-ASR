from tqdm import tqdm
import pandas as pd
import gc

import logging
from nemo.utils import logging as nemo_logging

nemo_logging.setLevel(logging.CRITICAL)

import torch
from faster_whisper import WhisperModel
from nemo.collections.asr.models import EncDecMultiTaskModel


class WhisperHelper:
    def __init__(self):
        pass

    def init_model(self, model_size="large-v3"):
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")

    def transcribe(self, path):
        segments, _ = self.model.transcribe(path, beam_size=5, language="en")
        return "".join([segment.text for segment in segments])

    def __str__(self):
        return "Wisper"

    def del_model(self):
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
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


df = pd.read_csv("processed_dataset.csv", sep=";")
file_paths = df["path"].to_list()

models = [WhisperHelper(), CanaryHelper()]

for model in models:
    model.init_model()

    transcripts = [None] * len(file_paths)
    for i, path in tqdm(enumerate(file_paths), total=len(file_paths)):
        transcripts[i] = model.transcribe(path)
    df[str(model)] = transcripts

    model.del_model()

df.to_csv("STT_output.csv", sep=";", index=False)
