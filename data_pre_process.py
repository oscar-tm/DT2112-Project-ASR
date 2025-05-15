###
#
#   Written by:
#   Oscar Tegnér Mohringe
#
#
#   Loads the data from provided common voice delta segment(s) and
#   corrupts the data with random noise sampled from the demand dataset.
#   Also downsamples the audio to 16kHz mono inorder for CMU Sphinx to
#   work properly.
#
###

from pydub import AudioSegment
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

np.random.seed(1234)
current_dir = os.getcwd() + "/"

paths = [
    "cv-corpus-10.0-delta-2022-07-04/en/",
]

filenames = ["validated.tsv"] * len(paths)

df = pd.DataFrame()
for f_path, filename in zip(paths, filenames):
    filepath = os.path.join(f_path, filename)
    temp_df = pd.read_csv(filepath, sep="\t")
    temp_df["folder_path"] = f_path + "clips/"
    df = pd.concat([df, temp_df], ignore_index=True)
    del temp_df

noise_cats = ["CLEAN", "SCAFE", "STRAFFIC", "TBUS", "DLIVING", "DKITCHEN"]

# Split the data into len(noise_cats) parts randomly
indices = np.arange(len(df))
np.random.shuffle(indices)

n_chunks = len(noise_cats)
chunk_size = len(df) // n_chunks
rem = len(df) % n_chunks

chunks = []
start = 0
# It is slow to iterate over dataframes
# -> We extract the relevant fields and
# put them in a list of tuples.
for i in range(n_chunks):
    end = start + chunk_size + (1 if i < rem else 0)
    chunk_indices = indices[start:end]
    df_chunk = df.iloc[chunk_indices].reset_index(drop=True)
    chunk_paths = df_chunk["path"].to_list()
    chunk_folder_paths = df_chunk["folder_path"].to_list()
    chunk_sentence = df_chunk["sentence"].to_list()
    chunk_ages = df_chunk["age"].to_list()
    chunk_accents = df_chunk["accents"].to_list()
    chunk_gender = df_chunk["gender"].to_list()
    chunks.append(
        list(
            zip(
                chunk_paths,
                chunk_folder_paths,
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
tot_time = 0.0

for noise_type in noise_cats:
    files = chunks[noise_cats.index(noise_type)]
    for f_path, folder_path, true_sentence, age, accent, gender in tqdm(files):
        # Set the the number of channels to 1 in order to make sure we are working with mono audio
        # We also need to downsample the audio from 32 kHz to 16 kHz inorder for CMU to work properly
        sentence = (
            AudioSegment.from_file(folder_path + f_path)
            .set_channels(1)
            .set_frame_rate(16000)
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

        # Create the testdata directory if it doesnt exist.
        if mkdir:
            os.makedirs(current_dir + path, exist_ok=True)
            mkdir = False

        # Save the files as wav instead of mp3
        relative_path = path + noise_type + "_" + f_path[:-4] + ".wav"

        sentence.export(
            current_dir + relative_path,
            format="wav",
        )

        df.append(
            (
                relative_path,
                sentence.duration_seconds,
                noise_type,
                true_sentence,
                age,
                accent,
                gender,
            )
        )
        tot_time += sentence.duration_seconds

# Name the columns of the new data frame
df = pd.DataFrame(
    df,
    columns=["path", "Length (s)", "Noise type", "sentence", "age", "accent", "gender"],
)
df.to_csv("processed_dataset.csv", index=False, sep=";")

print("Total time (h):", str(tot_time / (60 * 60)))
