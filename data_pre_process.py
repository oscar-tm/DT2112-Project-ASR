from pydub import AudioSegment
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

np.random.seed(1234)
current_dir = os.getcwd()
SNRS = [0, 5, 15, 25]

# Load the tsv from the dataset
df = pd.read_csv(
    "cv-corpus-20.0-delta-2024-12-06/en/validated_sentences.tsv", sep="\t"
)
ids = set(df["sentence_id"])

df = pd.concat(
    [
        pd.read_csv("cv-corpus-20.0-delta-2024-12-06/en/validated.tsv", sep="\t"),
        pd.read_csv("cv-corpus-20.0-delta-2024-12-06/en/other.tsv", sep="\t"),
    ],
    ignore_index=True,
)

noise_cats = ["CLEAN", "SCAFE", "STRAFFIC", "TBUS", "DLIVING", "DKITCHEN"]

### Split the data into len(noise_cats) parts randomly
indices = np.arange(len(df))
np.random.shuffle(indices)

n_chunks = len(noise_cats)
chunk_size = len(df) // n_chunks
rem = len(df) % n_chunks

chunks = []
start = 0
for i in range(n_chunks):
    end = start + chunk_size + (1 if i < rem else 0)
    chunk_indices = indices[start:end]
    df_chunk = df.iloc[chunk_indices].reset_index(drop=True)
    chunk_ids = df_chunk["sentence_id"].to_list()
    chunk_paths = df_chunk["path"].to_list()
    chunks.append(list(zip(chunk_ids, chunk_paths)))
    start = end

for noise_type in noise_cats:
    files = chunks[noise_cats.index(noise_type)]

    mkdir = True
    i = 0
    for sentence_id, f_path in tqdm(files):
        if sentence_id not in ids:  # We only want to use validated files
            continue
        i += 1
        if i > 100:
            break

        # Need to check if everything is in mono or stereo!! if stereo collapse to mono
        sound_path = "cv-corpus-20.0-delta-2024-12-06/en/clips/" + f_path
        sentence = AudioSegment.from_file(sound_path)

        if noise_type != "CLEAN":
            file_number = np.random.randint(low=1, high=17)

            # Load the noise
            noise = AudioSegment.from_file(
                "demand/SCAFE/ch"
                + (str(file_number) if file_number >= 10 else "0" + str(file_number))
                + ".wav"
            )

            # Get a random start point for the noise

            noise = noise[np.random.randint(low=0, high=len(noise) - len(sentence)) :]

            cSNR = sentence.dBFS - noise.dBFS

            # Set noise to same level as sentence
            noise = noise + cSNR

            for SNR in SNRS:
                # Modify noise sound level not speaker sound in order to avoid clipping etc.
                noise_mod = noise - SNR

                # Overlay the noise. The file will be as long as the orginal recording as over doesnt change the length of the base object
                overlay = sentence.overlay(noise_mod)

                path = (
                    "/testdata/" + noise_type + "/" + "SNR" + str(SNR) + "DB" + "/"
                )

                if mkdir:
                    os.makedirs(current_dir + path, exist_ok=True)

                overlay.export(current_dir + path + f_path[:-4] + ".wav", format="wav")

            mkdir = False

        else:
            path = "/testdata/" + noise_type + "/"

            if mkdir:
                os.makedirs(current_dir + path, exist_ok=True)
                mkdir = False

            sentence.export(current_dir + path + f_path[:-4] + ".wav", format="wav")
