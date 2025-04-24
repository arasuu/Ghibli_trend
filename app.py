import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Load the model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe.to(device)

# Streamlit UI
st.title("🎨 Ghibli-style Image Generator")
prompt = st.text_input("Enter a prompt:")

if st.button("Generate"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating image..."):
            image = pipe(prompt).images[0]
            st.image(image, caption="Generated Image")
