import base64
import io

import librosa
import modal
import numpy as np
import requests
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.transforms as T
from pydantic import BaseModel

from model import AudioCNN

app = modal.App("audio-cnn-inference")

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .apt_install(["libsndfile1"])
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
    

class InferenceRequest(BaseModel):
    audio_data: str
    

@app.cls(image=image, gpu="A10G", volumes={"/models": model_volume}, scaledown_window=15)
class AudioClassifier:
    @modal.enter()
    def load_model(self):
        print("Loading model on enter")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load("/models/best_model.pth", map_location=self.device)

        self.categories = checkpoint["categories"]
        self.model = AudioCNN(num_classes=len(self.categories))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.audio_processor = AudioProcessor()
        print("Model loaded on enter")

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: InferenceRequest):
        audio_bytes = base64.b64decode(request.audio_data)
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        # Convert audio data to one channel only
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        if sample_rate != 44100:
            audio_data = librosa.resample(
                y=audio_data, orig_sr=sample_rate, target_sr=44100
            )

        spectrogram = self.audio_processor.process_audio_chunk(audio_data)
        spectrogram = spectrogram.to(self.device)

        with torch.no_grad():
            output, feature_maps = self.model(spectrogram, return_feature_maps=True)
            output = torch.nan_to_num(output)

            # dim=0 batch, dim=1 category
            probabilities = torch.softmax(output, dim=1)
            top3_probabilities, top3_indices = torch.topk(probabilities[0], 3)

            predictions = [
                {"category": self.categories[index.item()], "confidence": probability.item()} 
                for probability, index in zip(top3_probabilities, top3_indices)
            ]

            visualization_data = {}
            for name, tensor in feature_maps.items():
                # [batch_size, channels, height, width]
                if tensor.dim() == 4:
                    # Convert to one channel only by taking the mean
                    aggregated_tensor = torch.mean(tensor, dim=1)

                    # Remove batch_size from tensor
                    squeezed_tensor = aggregated_tensor.squeeze(0)

                    numpy_array = squeezed_tensor.cpu().numpy()
                    clean_array = np.nan_to_num(numpy_array)
                    visualization_data[name] = {
                        "shape": list(clean_array.shape),
                        "values": clean_array.tolist()
                    }

            # Squeeze the first two dimensions
            # [batch, channel, height, width] -> [height, width]
            spectrogram_numpy = spectrogram.squeeze(0).squeeze(0).cpu().numpy()
            clean_spectrogram = np.nan_to_num(spectrogram_numpy)

            max_samples = 8000
            if len(audio_data) > max_samples:
                step = len(audio_data) // max_samples
                waveform_data = audio_data[::step]
            else:
                waveform_data = audio_data

        response = {
            "predictions": predictions,
            "visualization": visualization_data,
            "input_spectrogram": {
                "shape": list(clean_spectrogram.shape),
                "values": clean_spectrogram.tolist()
            },
            "waveform": {
                "values": waveform_data.tolist(),
                "sample_rate": 44100,
                "duration": len(audio_data) / 44100
            }
        }

        return response
    

@app.local_entrypoint()
def main():
    audio_data, sample_rate = sf.read("chirpingbirds.wav")
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    payload = {"audio_data": audio_b64}

    server = AudioClassifier()
    url = server.inference.get_web_url()
    response = requests.post(url, json=payload)
    response.raise_for_status()

    result = response.json()

    waveform_info = result.get("waveform", {})
    if waveform_info:
        values = waveform_info.get("values", {})
        print(f"First 10 values: {[round(value, 4) for value in values[:10]]}...")
        print(f"Duration: {waveform_info.get("duration", 0)}")

    print("Top predictions:")
    for prediction in result.get("predictions", []):
        print(f" -{prediction["category"]} {prediction["confidence"]:0.2%}")