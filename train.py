import sys
from datetime import datetime
from pathlib import Path

import modal
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import AudioCNN

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
            spectrogram = self.transform(waveform)
        else:
            spectrogram = waveform

        return spectrogram, row["label"]
    

def mixup_data(x, y):
    percentage = np.random.beta(0.2, 0.2)
    batch_size = x.size(0)

    # Shuffle the batch of audio clips
    index = torch.randperm(batch_size).to(x.device)

    # Create a new sample based on two audio clips
    # e.g. (0.7 * audio1) + (0.3 * audio2)
    mixed_x = (percentage * x) + ((1 - percentage) * x[index, :])
    y_label1, y_label2 = y, y[index]

    return mixed_x, y_label1, y_label2, percentage


def mixup_criterion(criterion, predicted, y_label1, y_label2, percentage):
    return (percentage * criterion(predicted, y_label1) + 
            (1 - percentage) * criterion(predicted, y_label2))


@app.function(image=image, gpu="A10G", volumes={"/data": volume, "/models": model_volume}, timeout=60 * 60 * 3)
def train():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"/models/tensorboard_logs/run_{timestamp}"
    writer = SummaryWriter(log_dir)

    esc50_dir = Path("/opt/esc50-data")

    # Turn the WAV file into a spectrogram
    train_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050, 
            n_fft=1024, 
            hop_length=512, 
            n_mels=128, 
            f_min=0, 
            f_max=11025
        ),
        T.AmplitudeToDB(),
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=80)
    )

    validation_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050, 
            n_fft=1024, 
            hop_length=512, 
            n_mels=128, 
            f_min=0, 
            f_max=11025
        ),
        T.AmplitudeToDB()
    )

    train_dataset = ESC50Dataset(
        data_dir=esc50_dir, 
        metadata_file=esc50_dir / "meta" / "esc50.csv", 
        split_train=True, 
        transform=train_transform
    )

    validation_dataset = ESC50Dataset(
        data_dir=esc50_dir, 
        metadata_file=esc50_dir / "meta" / "esc50.csv", 
        split_train=False, 
        transform=validation_transform
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(num_classes=len(train_dataset.categories))
    model.to(device)

    num_epochs = 100
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)

    # Change the learning rate after every batch
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=0.002, 
        epochs=num_epochs, 
        steps_per_epoch=len(train_dataloader), 
        pct_start=0.1
    )

    best_accuracy = 0.0

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for data, label in progress_bar:
            # Move variables to GPU if available
            data, label = data.to(device), label.to(device)

            # Combine two audio files 30% of the time
            if np.random.random() > 0.7:
                data, label1, label2, percentage = mixup_data(data, label)
                output = model(data)
                loss = mixup_criterion(criterion, output, label1, label2, percentage)
            else:
                output = model(data)
                loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        writer.add_scalar("Loss/Train", avg_epoch_loss, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)

        # Validation after each epoch
        model.eval()

        correct = 0
        total = 0
        validation_loss = 0

        with torch.no_grad():
            for data, label in test_dataloader:
                data, label = data.to(device), label.to(device)
                output = model(data)
                loss = criterion(output, label)
                validation_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        accuracy = 100 * correct / total
        avg_validation_loss = validation_loss / len(test_dataloader)
        writer.add_scalar("Loss/Validation", avg_validation_loss, epoch)
        writer.add_scalar("Accuracy/Validation", accuracy, epoch)

        print(f"Epoch {epoch + 1}: Loss: {avg_epoch_loss:.4f}%, "
              f"Validation Loss: {avg_validation_loss:.4f}%, Accuracy: {accuracy:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                "model_state_dict": model.state_dict(),
                "accuracy": accuracy,
                "epoch": epoch,
                "categories": train_dataset.categories
            }, "/models/best_model.pth")
            print(f"New best model saved: {accuracy:.2f}%")

    writer.close()
    print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")


@app.local_entrypoint()
def main():
    train.remote()