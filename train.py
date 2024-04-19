import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from whisper.models import Transformer, TransformerConfig
from whisper.data import KathbathDataset, collate_batch

# Define training parameters
batch_size = 16
num_epochs = 10
learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model configuration
config = TransformerConfig(
    vocab_size=10000,  # Adjust based on your vocabulary size
    d_model=512,
    num_encoder_layers=6,
    num_decoder_layers=6,
    num_heads=8,
    d_ff=2048,
    dropout=0.1
)

# Initialize model
model = Transformer(config).to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the padding token

# Load and preprocess Kathbath dataset
train_dataset = KathbathDataset(data_dir='path_to_preprocessed_data/train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):
        spectrograms, targets = batch
        spectrograms, targets = spectrograms.to(device), targets.to(device)
        
        optimizer.zero_grad()
        output = model(spectrograms)
        
        # Calculate loss
        loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

# Save trained model
torch.save(model.state_dict(), 'whisper_model.pth')