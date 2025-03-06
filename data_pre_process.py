from pydub import AudioSegment
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

np.random.seed(1234)
current_dir = os.getcwd() + "/"

# Load the validated tsv from the dataset
df = pd.read_csv("cv-corpus-10.0-delta-2022-07-04/en/validated.tsv", sep="\t")

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
    chunk_paths = df_chunk["path"].to_list()
    chunk_sentence = df_chunk["sentence"].to_list()
    chunk_ages = df_chunk["age"].to_list()
    chunk_accents = df_chunk["accents"].to_list()
    chunk_gender = df_chunk["gender"].to_list()
    chunks.append(
        list(
            zip(
                chunk_paths,
                chunk_sentence,
                chunk_ages,
                chunk_accents,
                chunk_gender,
            )
        )
    )
    start = end

df = []
mkdir = True
for noise_type in noise_cats:
    files = chunks[noise_cats.index(noise_type)]

    for f_path, true_sentence, age, accent, gender in tqdm(files):
        sound_path = "cv-corpus-10.0-delta-2022-07-04/en/clips/" + f_path
        # Set the the number of channels to 1 in order to make sure we are working with mono audio
        # We also need to downsample the audio from 32 kHz to 16 kHz inorder for CMU to work properly
        sentence = (
            AudioSegment.from_file(sound_path).set_channels(1).set_frame_rate(16000)
        )

        if noise_type != "CLEAN":
            file_number = np.random.randint(low=1, high=17)

            # Load the noise
            noise = (
                AudioSegment.from_file(
                    "demand/SCAFE/ch"
                    + (
                        str(file_number)
                        if file_number >= 10
                        else "0" + str(file_number)
                    )
                    + ".wav"
                )
                .set_channels(1)
                .set_frame_rate(16000)
            )

            # Get a random start point for the noise

            noise = noise[np.random.randint(low=0, high=len(noise) - len(sentence)) :]
            cSNR = sentence.dBFS - noise.dBFS

            # Set noise to same level as sentence
            noise = noise + cSNR

            # Overlay the noise. The file will be as long as the orginal recording as over doesnt change the length of the base object
            sentence = sentence.overlay(noise)

        path = "testdata/"

        if mkdir:
            os.makedirs(current_dir + path, exist_ok=True)
            mkdir = False

        relative_path = path + noise_type + "_" + f_path[:-4] + ".wav"

        sentence.export(
            current_dir + relative_path,
            format="wav",
        )

        df.append((relative_path, noise_type, true_sentence, age, accent, gender))

df = pd.DataFrame(
    df, columns=["path", "Noise type", "sentence", "age", "accent", "gender"]
)
df.to_csv("processed_dataset.csv", index=False, sep=";")
