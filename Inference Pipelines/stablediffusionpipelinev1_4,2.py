# -*- coding: utf-8 -*-
"""StableDiffusionPipelinev1-4,2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1UQKQ2INHSzpmgXsQJT5SFKSRz_Xn1UrR
"""

# Necessary libraries to import
import torch
from diffusers import StableDiffusionPipeline

from google.colab import drive
drive.mount('/content/drive')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = None

# Loading the pretrained stable diffusion model named "CompVis/stable-diffusion-v1-4".
if device == "cuda":
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
else:
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

pipe = pipe.to(device)

prompt = "A background image to be used in the advertisement of a Coca-Cola."
high_guidance_scale_value = 15  # A larger guidance scale for more fidelity to the prompt

# Generating the image with the adjusted guidance scale
image = pipe(prompt, guidance_scale=high_guidance_scale_value).images[0]
image.save("background_with_high_guidance.png")
print(f"Image generated with guidance_scale={high_guidance_scale_value} and saved to background_with_high_guidance.png.")

low_guidance_scale_value = 5
image = pipe(prompt, guidance_scale=low_guidance_scale_value).images[0]
image.save("background_with_low_guidance.png")
print(f"Image is generated with guidance_scale={low_guidance_scale_value} and saved to background_with_low_guidance.png.")

image = pipe(prompt).images[0]
image.save('background.png')
print("Background image is generated and saved to background.png.")

prompt = "A bearded man is smoking and walking in a Turkish movie from the Yesilcam era."
high_guidance_scale_value = 15  # A larger guidance scale for more fidelity to the prompt

# Generating the image with the adjusted guidance scale
image = pipe(prompt, guidance_scale=high_guidance_scale_value).images[0]
image.save("background_with_high_guidance_yesilcam.png")
print(f"Image is generated with guidance_scale={high_guidance_scale_value}, and saved to background_with_high_guidance_yesilcam.png.")

image = pipe(prompt, guidance_scale=low_guidance_scale_value, num_inference_steps=100)
image.save("background_with_high_number_of_inference_steps")
print(f"Image is generated with num_inference_steps = 100 and guidance_scale={low_guidance_scale}, and saved to background_with_high_number_of_inference_steps.png.")

image = pipe(prompt, guidance_scale=low_guidance_scale_value, num_inference_steps=10)
image.save("background_with_low_number_of_inference_steps")
print(f"Image is generated with num_inference_steps = 10 and guidance_scale={low_guidance_scale}, and saved to background_with_low_number_of_inference_steps.png.")









