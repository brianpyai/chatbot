"""
!pip install -q diffusers transformers accelerate peft 
!pip install --upgrade accelerate
"""



import gc, time, os, sys, json
from pathlib import Path
from tqdm.auto import tqdm

gc.collect()

try:
    import google.colab
    IN_COLAB = True
    from google.colab import drive
    drive.mount('/gdrive')
    Gbase = "/gdrive/MyDrive/generate/"
    cache_dir = "/gdrive/MyDrive/hf/"
    sys.path.append(Gbase)
except:
    IN_COLAB = False
    Gbase = "./generate/"
    cache_dir = "./hf/"

import torch
from torch.utils.data import Dataset
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusionXLPipeline
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
import numpy as np

# Initialize the Accelerator first
accelerator = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else "no")

# Set a seed for reproducibility
accelerator.wait_for_everyone()
set_seed(42)

model_id= "stabilityai/stable-diffusion-xl-base-1.0"

def listImages(d):
    images = []
    for f in os.scandir(d):
        if f.is_file() and f.name.split(".")[-1].lower() in ("jpg", "jpeg", "png", "bmp", "svg"):
            images.append(f.path)
    return images

def imageResizeX64(image):
    w, h = image.size
    new_width = int(w // 64) * 64
    new_height = int(h // 64) * 64
    new_image = Image.new("RGB", (new_width, new_height))
    offset_x = int((new_width - w) / 2)
    offset_y = int((new_height - h) / 2)
    new_image.paste(image, (offset_x, offset_y))
    return new_image



def load_image_to(image_path, max_size=768):
    if isinstance(image_path, Image.Image):
        image = image_path.convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    
    # Resize to a multiple of 64
    width, height = image.size
    new_width = (width // 64) * 64
    new_height = (height // 64) * 64
    
    image = image.resize((new_width, new_height))
    return image


class CustomDataset(Dataset):
    def __init__(self, image_paths, prompts):
        self.image_paths = image_paths
        self.prompts = prompts
        self.max_size = 1024

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_image_to(self.image_paths[idx], self.max_size)
        prompt = self.prompts[idx]
        
        # Convert image to tensor and normalize to [-1, 1]
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image = (image - 0.5) * 2
        
        return {
            "pixel_values": image,
            "prompt": prompt,
        }


from peft import get_peft_model_state_dict
save_path = os.path.join(Gbase, "lora_weights_epoch.pt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load the pipeline with the appropriate data type
pipeline = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
pipeline.to(accelerator.device)

# Determine target modules for LoRA
target_modules = ["to_q", "to_v"]

lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=target_modules, lora_dropout=0.05, bias="none")

# Apply LoRA to UNet
pipeline.unet = get_peft_model(pipeline.unet, lora_config)

if os.path.exists(save_path):
    print(f"Loading existing LoRA weights from {save_path}")
    state_dict = torch.load(save_path, map_location=device)
    pipeline.unet.load_state_dict(state_dict, strict=False)
    print("LoRA weights loaded successfully")
else:
    print("No existing LoRA weights found. Starting training from scratch.")

# Prepare the dataset
image_paths = listImages(os.path.join(Gbase, "newRef"))
tempPrompts = {}
image_paths1 = []
with open(os.path.join(Gbase, "allPrompts.json"), 'r', encoding='utf-8') as f:
    tempPrompts = json.load(f)
prompts = []
for p in image_paths:
    k = Path(p).name 
    if k in tempPrompts:
        prompts.append(tempPrompts[k])
        image_paths1.append(p)

dataset = CustomDataset(image_paths1, prompts)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# Create optimizer
optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=1e-4)

# Prepare models and data for training
pipeline.unet, optimizer, dataloader = accelerator.prepare(
    pipeline.unet, optimizer, dataloader
)

# ... (previous code remains the same)

pipeline.unet.train()
num_epochs = 1
total_steps = len(dataloader) * num_epochs
progress_bar = tqdm(total=total_steps, desc="Training")



# Add this function to save the LoRA weights
def save_lora_weights(pipeline, epoch, step, save_path):
    lora_state_dict = get_peft_model_state_dict(pipeline.unet)
    torch.save(lora_state_dict, _use_new_zipfile_serialization=False)
    print(f"LoRA weights saved at epoch {epoch}, step {step}")


for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        with accelerator.accumulate(pipeline.unet):
            # Move inputs to the correct device and adjust data types
            pixel_values = batch['pixel_values'].to(accelerator.device, dtype=pipeline.vae.dtype, non_blocking=True)
            prompts = batch['prompt']

            # Tokenize prompts with tokenizer 1 (for text_encoder)
            encodings = pipeline.tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
                return_tensors="pt",
            )
            input_ids = encodings.input_ids.to(accelerator.device)
            attention_mask = encodings.attention_mask.to(accelerator.device)

            # Tokenize prompts with tokenizer 2 (for text_encoder_2)
            encodings_2 = pipeline.tokenizer_2(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer_2.model_max_length,
                return_tensors="pt",
            )
            input_ids_2 = encodings_2.input_ids.to(accelerator.device)
            attention_mask_2 = encodings_2.attention_mask.to(accelerator.device)

            # Encode images to latents
            latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * pipeline.vae.config.scaling_factor

            # Prepare noise and add to latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

            # Get text embeddings
            prompt_embeds = pipeline.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            prompt_embeds = prompt_embeds.hidden_states[-2]

            # Get pooled text embeddings
            pooled_prompt_embeds = pipeline.text_encoder_2(
                input_ids=input_ids_2,
                attention_mask=attention_mask_2,
                output_hidden_states=True,
            )
            pooled_prompt_embeds = pooled_prompt_embeds.last_hidden_state
            text_embeds = pipeline.text_encoder_2.text_model.final_layer_norm(pooled_prompt_embeds.mean(dim=1))

            # Prepare time_ids
            add_time_ids = torch.zeros((bsz, 6), device=latents.device)

            # Print shapes for debugging
           # print(f"noisy_latents shape: {noisy_latents.shape}")
           # print(f"prompt_embeds shape: {prompt_embeds.shape}")
            #print(f"text_embeds shape: {text_embeds.shape}")
           # print(f"add_time_ids shape: {add_time_ids.shape}")

            # Adjust dimensions of prompt_embeds
            if prompt_embeds.shape[-1] != 2048:
                prompt_embeds = torch.cat([prompt_embeds, torch.zeros(*prompt_embeds.shape[:-1], 2048 - prompt_embeds.shape[-1], device=prompt_embeds.device)], dim=-1)

            # Get model output
            model_output = pipeline.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": add_time_ids},
                return_dict=False,
            )[0]

            # Compute loss
            loss = torch.nn.functional.mse_loss(model_output.float(), noise.float(), reduction="mean")

            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        progress_bar.update(1)
        progress_bar.set_postfix({"loss": loss.item(), "epoch": epoch + 1})
        saveN=1000 if torch.cuda.is_available() else 150
        if (step + 1) % saveN == 0:save_lora_weights(pipeline, epoch, step, save_path)

progress_bar.close()
final_save_path = os.path.join(Gbase, "final_lora_weights.pt")
save_lora_weights(pipeline, num_epochs - 1, total_steps - 1, final_save_path)