import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from PIL import Image
import os
import random


# Set up the Stable Diffusion model pipeline
def setup_pipeline(model_name="stabilityai/stable-diffusion-2-1-base"):
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    # Explicitly check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # If no GPU (CUDA) is available, fallback to CPU
    if device.type == "cpu":
        print("CUDA not available. Dataset will be created on CPU.")
        
    pipe.to(device) 
    return pipe



def generate_face_pair(pipe, height=512, width=512):
    # Detailed description of the same person with more precision to ensure consistency
    person_description = "A man with short brown hair, light skin, a strong jawline, and facial features typical of a man in his 40s"

    # Generate the clean-shaven version of the person (no beard)
    clean_shaven_prompt = f"{person_description}, clean-shaven"
    clean_shaven_image = pipe(clean_shaven_prompt, height=height, width=width).images[0]

    # Generate the bearded version of the same person (with beard)
    bearded_prompt = f"{person_description}, with a full beard"
    bearded_image = pipe(bearded_prompt, height=height, width=width).images[0]
    
    return clean_shaven_image, bearded_image



def generate_dataset(pipe, num_pairs=100, output_dir="dataset/generated_faces"):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_pairs):
        clean_shaven_image, bearded_image = generate_face_pair(pipe)
        
        # Save the images
        clean_shaven_image.save(f"{output_dir}/clean_shaven/clean_shaven_{i+1}.png")
        bearded_image.save(f"{output_dir}/bearded/bearded_{i+1}.png")

        # Optionally, print progress
        if i % 10 == 0:
            print(f"Generated {i+1} pairs...")
    
    print(f"Dataset generated with {num_pairs} pairs.")




def preview_images(output_dir, num_images=10):
    # List all the clean-shaven and bearded images
    clean_shaven_images = [f"{output_dir}/clean_shaven/clean_shaven_{i+1}.png" for i in range(num_images)]
    bearded_images = [f"{output_dir}/bearded/bearded_{i+1}.png" for i in range(num_images)]
    
    # Plot the images in pairs
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 50))
    
    for i in range(num_images):
        # Open and show the clean-shaven image
        clean_image = Image.open(clean_shaven_images[i])
        axes[i, 0].imshow(clean_image)
        axes[i, 0].axis('off')
        
        # Open and show the bearded image
        bearded_image = Image.open(bearded_images[i])
        axes[i, 1].imshow(bearded_image)
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
