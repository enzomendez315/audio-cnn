import sys
from pathlib import Path

import modal
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

app = modal.App("audio-cnn")

# Build Docker image
image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"])
         .run_commands([
             "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
             "cd /tmp && unzip esc50.zip",
             "mkdir -p /opt/esc50-data",
             "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
             "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
         ])
         .add_local_python_source("model"))

volume = modal.Volume.from_name("esc50-data", create_if_missing=True) # Attaches to the data
model_volume = modal.Volume.from_name("esc-model", create_if_missing=True) # Attaches to the model


class ESC50Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, split_train=True, transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.split_train = split_train
        self.transform = transform

        # Filter rows by the "fold" column
        if split_train:
            # Training - 80% of dataset
            self.metadata = self.metadata[self.metadata["fold"] != 5]
        else:
            # Validation - 20% of dataset
            self.metadata = self.metadata[self.metadata["fold"] == 5]

        # Add numeric values to the categories
        self.categories = sorted(self.metadata["category"].unique())
        self.category_labels = {category: index for index, category in enumerate(self.categories)}
        self.metadata["label"] = self.metadata["category"].map(self.category_labels)

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        audio_path = self.data_dir / "audio" / row["filename"]
        waveform, sample_rate = torchaudio.load(audio_path)

        # Reduce the number of channels [waveform, samples] to one channel by averaging
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.transform:
            spectogram = self.transform(waveform)
        else:
            spectogram = waveform

        return spectogram, row["label"]


@app.function(image=image, gpu="A10G", volumes={"/data": volume, "/models": model_volume}, timeout=60 * 60 * 3)
def train():
    print("Training")


@app.local_entrypoint()
def main():
    train.remote()