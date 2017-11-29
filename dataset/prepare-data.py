# Run this script inside dataset for one genre (e.g. classical)
# Gets all mp3 files, converts them to wav then creates a spectrogram


import fnmatch
import os
import subprocess

ROOT = "/Users/jamieg/Documents/Work/Hackday/tensorflow-stuff-2/"

# Go inside all directories and find all mp3 files
mp3_files = []
for root, dirnames, filenames in os.walk(ROOT + "prepare-data/classical/mp3s"):
    for filename in fnmatch.filter(filenames, '*.mp3'):
        mp3_files.append(os.path.join(root, filename))


# Convert all mp3 files into wav files and save them in wav_files directory
for i in range(len(mp3_files)):
    outputFilename = ROOT + "prepare-data/classical/wav_files/output-" + str(i) + ".wav"
    subprocess.call(["ffmpeg", "-i", mp3_files[i], outputFilename])


# Create spectrogram from each wav file and save in spectrograms directory