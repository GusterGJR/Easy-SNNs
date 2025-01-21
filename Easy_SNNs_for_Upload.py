from parameters import *
import torch
import torch.nn as nn
import torchaudio
from spikingjelly.clock_driven import neuron, encoding, functional
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

losses = []
predicts = []

class AudioDurationPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AudioDurationPredictor, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.spike_neuron = neuron.LIFNode()
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.spike_neuron(x)
        x = self.decoder(x)
        return x
    
def preprocess_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    
    if waveform.size(1) < sample_rate * seconds:
        raise ValueError(f"Audio file is shorter than {seconds} seconds!")

    first_two_seconds = waveform[:, :int(sample_rate * seconds)]
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(first_two_seconds)
    features = mel_spectrogram.mean(dim = 0).flatten()
    return features

def predict_duration(model, audio_path):
    model.eval()
    with torch.no_grad():
        features = preprocess_audio(audio_path)
        features = features.unsqueeze(0)
        prediction = model(features)

        return prediction

def test_model(model):
    predicted_durations = predict_duration(model, test_audio_path)
    predicted_duration = predicted_durations[0].item()
    print(f"Predicted Duration: {predicted_duration:.2f} seconds") # Actual Duration: 164.00 seconds
    return predicted_duration

def train_model(model, data_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward(retain_graph=True)
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
        
        losses.append(loss.item())
        predicts.append(test_model(model = model))
        
        print()
    
def create_dataset(audio_paths, durations):
    features = [preprocess_audio(path) for path in audio_paths]
    inputs = torch.stack(features)
    targets = torch.tensor(durations).float()
    return TensorDataset(inputs, targets)

def visualize_training(epochs):
    fig, ax1 = plt.subplots(figsize = (10, 6))
    ax1.plot(range(1, epochs + 1), losses, label = "Loss", color = "blue")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color = "blue")
    ax1.tick_params(axis = "y", labelcolor = "blue")
    
    ax2 = ax1.twinx()
    ax2.plot(range(1, epochs + 1), predicts, label = "Predicted Duration", color = "red")
    ax2.set_ylabel("Predicted Duration (s)", color = "red")
    ax2.tick_params(axis = "y", labelcolor = "red")
    
    plt.title("Training Loss and Predicted Duration")
    plt.grid(alpha = 0.7)
    plt.show()
    
if __name__ == "__main__":
    model = AudioDurationPredictor(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    audio_paths = [audio1, audio2, audio3]
    durations = [263, 271, 215] # audio seconds

    # Create DataLoader for training
    dataset = create_dataset(audio_paths, durations)
    data_loader = DataLoader(dataset, batch_size, shuffle = True)

    # Train the model
    train_model(model, data_loader, criterion, optimizer, epochs)
    
    visualize_training(epochs)