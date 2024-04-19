import torch

from torch.utils.data import DataLoader
from whisper.data import KathbathDataset, collate_batch
from whisper.utils import decode

# Load trained model
model = Transformer(config).to(device)  # Assuming config is defined as before
model.load_state_dict(torch.load('whisper_model.pth'))
model.eval()

# Load test dataset
test_dataset = KathbathDataset(data_dir='path_to_preprocessed_data/test')
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_batch)

total_wer = 0
total_words = 0

# Evaluation loop
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        spectrograms, targets = batch
        spectrograms, targets = spectrograms.to(device), targets.to(device)
        
        # Perform inference
        output = model(spectrograms)
        
        # Decode predicted output
        predicted_transcriptions = decode(output.argmax(dim=-1))
        
        # Calculate WER for this batch
        for pred, target in zip(predicted_transcriptions, targets):
            total_wer += wer(pred, target)
            total_words += len(target.split())
    
# Calculate overall WER
overall_wer = total_wer / total_words
print(f'Overall Word Error Rate (WER): {overall_wer:.4f}')