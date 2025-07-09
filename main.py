import modal
import torch
import torch.nn as nn
import torchaudio.transforms as T

app = modal.App("audio-cnn-inference")

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .add_local_python_source("model"))

model_volume = modal.Volume.from_name("esc-model")


class AudioProcessor:
    def __init__(self):
        self.transform = nn.Sequential(
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

    def process_audio_chunk(self, audio_data):
        # Convert array to torch tensor
        waveform = torch.from_numpy(audio_data).float()

        # Add another dimension to tensor
        waveform = waveform.unsqueeze(0)

        spectogram = self.transform(waveform)

        return spectogram.unsqueeze(0)
    

@app.cls(image=image, gpu="A10G", volumes={"/models": model_volume}, scaledown_window=15)
class AudioClassifier:
    @modal.enter()
    def load_model(self):
        pass