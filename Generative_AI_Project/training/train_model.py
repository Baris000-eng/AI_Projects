import torch
from torch.utils.data import DataLoader
import os
from models.unet import UNet
from torch.optim import Adam
from torchvision import transforms
from PIL import Image
import random

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, clean_shaven_dir, bearded_dir, transform=None):
        self.clean_shaven_images = sorted([os.path.join(clean_shaven_dir, f) for f in os.listdir(clean_shaven_dir)])
        self.bearded_images = sorted([os.path.join(bearded_dir, f) for f in os.listdir(bearded_dir)])
        self.transform = transform
    
    def __len__(self):
        return len(self.clean_shaven_images)
    
    def __getitem__(self, idx):
        clean_shaven_image = Image.open(self.clean_shaven_images[idx]).convert('RGB')
        bearded_image = Image.open(self.bearded_images[idx]).convert('RGB')
        
        if self.transform:
            clean_shaven_image = self.transform(clean_shaven_image)
            bearded_image = self.transform(bearded_image)
        
        return clean_shaven_image, bearded_image

def train_model(dataset_dir, epochs=10, batch_size=4, learning_rate=1e-4, model_save_dir='models'):
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1] range
    ])
    
    # Create dataset and dataloaders
    clean_shaven_dir = os.path.join(dataset_dir, 'clean_shaven')
    bearded_dir = os.path.join(dataset_dir, 'bearded')
    dataset = PairedDataset(clean_shaven_dir, bearded_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, optimizer, and loss function
    model = UNet(in_channels=3, out_channels=3).cuda()  # Move model to GPU if available
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.L1Loss()  # Use L1 loss for image-to-image translation
    
    # Ensure the model save directory exists
    os.makedirs(model_save_dir, exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (clean_shaven, bearded) in enumerate(dataloader):
            clean_shaven, bearded = clean_shaven.cuda(), bearded.cuda()
            
            optimizer.zero_grad()
            outputs = model(clean_shaven)
            loss = criterion(outputs, bearded)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i+1) % 100 == 0:  # Print every 100 steps
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {running_loss/100:.4f}")
                running_loss = 0.0
        
        # Save model checkpoint after every epoch in the `models/` directory
        checkpoint_path = os.path.join(model_save_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")
    
    print("Training completed!")


train_model(dataset_dir="dataset/generated_faces", model_save_dir="models")
