import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np

def validate_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for i, (bearded_images, clean_shaven_images) in enumerate(test_loader):
            bearded_images = bearded_images.cuda()
            clean_shaven_images = clean_shaven_images.cuda()

            # Forward pass through the model to get output
            outputs = model(bearded_images)

            # Visualize first 5 input-output pairs
            if i == 0:
                for j in range(5):
                    # Convert tensors back to images
                    bearded_image = bearded_images[j].cpu().numpy().transpose(1, 2, 0)
                    output_image = outputs[j].cpu().numpy().transpose(1, 2, 0)
                    clean_shaven_image = clean_shaven_images[j].cpu().numpy().transpose(1, 2, 0)

                    # Plot images
                    plt.figure(figsize=(12, 4))
                    plt.subplot(1, 3, 1)
                    plt.imshow(bearded_image)
                    plt.title("Bearded Image")
                    plt.axis('off')

                    plt.subplot(1, 3, 2)
                    plt.imshow(output_image)
                    plt.title("Model Output")
                    plt.axis('off')

                    plt.subplot(1, 3, 3)
                    plt.imshow(clean_shaven_image)
                    plt.title("Clean-Shaven Image")
                    plt.axis('off')

                    plt.show()


def compute_ssim(output, ground_truth):
    output = output.cpu().numpy().transpose(1, 2, 0)
    ground_truth = ground_truth.cpu().numpy().transpose(1, 2, 0)
    return ssim(output, ground_truth, multichannel=True)

# Example of SSIM calculation
ssim_value = compute_ssim(outputs[0], clean_shaven_images[0])
print(f"SSIM: {ssim_value}")


# Load the test data
test_dataset = BeardedDataset(clean_shaven_dir="dataset/generated_faces", 
                              bearded_dir="dataset/generated_faces", 
                              transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Validate the model
validate_model(model, test_loader)



