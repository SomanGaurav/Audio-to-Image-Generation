import os
import librosa
import tensorflow_hub as hub
import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from diffusers import StableDiffusion3Pipeline
import streamlit as st
import tempfile

# --- 1. Your Transformation Network Class (Unchanged) ---
class TransformationNetwork(nn.Module):
    def __init__(self , input_dim =128, output_dim = 2048):
        super(TransformationNetwork , self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim , 512),
            nn.GELU(),
            nn.LayerNorm(512),  # Add this
            nn.Dropout(0.3),
            nn.Linear(512 , 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024 , 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.4),
            nn.Linear(1024 , output_dim)
        )

    def forward(self , x):
        return self.layers(x)
# --- 2. Model & Data Loading (Cached for Speed) ---
# These functions run only ONCE.

class TransformationNetwork1(nn.Module):
    def __init__(self , input_dim =128, output_dim = 2048):
        super(TransformationNetwork1 , self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim , 512),
            nn.GELU(),
            nn.LayerNorm(512),  # Add this
            nn.Dropout(0.3),
            nn.Linear(512 , 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024 , 4096),
            nn.GELU(),
            nn.LayerNorm(4096),
            nn.Dropout(0.4),
            nn.Linear(4096 , output_dim)
        )

    def forward(self , x):
        return self.layers(x)

class TransformationNetwork2(nn.Module):
    def __init__(self , input_dim =128, output_dim = 2048):
        super(TransformationNetwork2 , self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim , 512),
            nn.GELU(),
            nn.LayerNorm(512),  # Add this
            nn.Dropout(0.2),
            nn.Linear(512 , 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024 , 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024 , 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            # nn.BatchNorm1d(512),  # Add this
            # nn.ReLU() ,
            # nn.Dropout(0.2),
            # nn.Linear(512 , 1024),
            # nn.BatchNorm1d(1024),  # Add this
            # nn.ReLU() ,
            # nn.Dropout(0.2),
            nn.Linear(2048 , output_dim)
        )

    def forward(self , x):
        return self.layers(x)
@st.cache_resource
def load_inference_data():
    """Loads your .npy database files."""
    st.write("Loading embedding database...")
    pooled_data = np.load("file4.npy")
    class_data = np.load("file2.npy")
    prompt_data = np.load("file1.npy")
    st.write("Database loaded.")
    return pooled_data, class_data, prompt_data

@st.cache_resource
def load_sd_pipeline():
    """Loads the Stable Diffusion 3 model onto GPU 1."""
    st.write("Loading Stable Diffusion 3 Pipeline (this takes a moment)...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
    ).to("cuda:0")
    st.write("Stable Diffusion 3 loaded.")
    return pipe

@st.cache_resource
def load_vggish_model():
    """Loads the VGGish model onto the CPU."""
    st.write("Loading VGGish audio model...")
    with tf.device('/cpu:0'):
        model = hub.load("https://tfhub.dev/google/vggish/1")
    st.write("VGGish model loaded.")
    return model

@st.cache_resource
def load_transformation_network():
    """Loads your trained audio-to-embedding model."""
    st.write("Loading V2D mapping network...")
    model = TransformationNetwork().to("cpu")
    state_dict = torch.load("best_model(3).pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    st.write("V2D mapping network loaded.")
    return model


def preprocess_audio(file_path):
    """Loads and preprocesses audio file."""
    waveform, sample_rate = librosa.load(file_path, sr=16000, mono=True)
    return waveform

def create_audio_encode(filepath, vggish_model):
    """
    Creates a VGGish embedding from an audio file.
    Takes the loaded model as an argument.
    """
    if not os.path.exists(filepath):
        st.error(f"Filepath does not exist: {filepath}")
        return None

    try:
        with tf.device('/cpu:0'):
            waveform_data = preprocess_audio(filepath)
            embeddings = vggish_model(waveform_data)
        
        mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
        mean_embedding = mean_embedding.reshape(1, 128)
        return mean_embedding.tolist()
    except Exception as e:
        st.error(f"Failed to produce embedding: {e}")
        return None




def main():
    st.set_page_config(page_title="Audio-to-Image Generator", layout="centered")
    st.title("Audio-to-Image Generator")
    st.write("Upload an audio file (.wav) to generate an image based on its sound.")

    # Load all models and data on startup
    try:
        pipe = load_sd_pipeline()
        vggish_model = load_vggish_model()
        transformer_model = load_transformation_network()
        # pooled_data, class_data, prompt_data = load_inference_data()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("Please ensure your models and .npy files are in the correct location and all dependencies are installed.")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

    if uploaded_file is not None:
        st.subheader("🎵 Uploaded Audio Preview")
        st.audio(uploaded_file, format="audio/wav")
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_filepath = tmp_file.name
        
        with st.spinner("Processing audio and generating image..."):
            try:
                audio_embedding = create_audio_encode(temp_filepath, vggish_model)
                if audio_embedding is None:
                    st.error("Could not process audio.")
                    return 

                data_tensor = torch.tensor(audio_embedding, dtype=torch.float32).to("cpu")
                with torch.no_grad():
                    output_embedding = transformer_model(data_tensor)
                output_numpy = output_embedding.numpy()
                blank_prompt_embeds = torch.tensor(
                    np.zeros([1, 333, 4096]), 
                    dtype=torch.float16,
                    device="cuda:0"
                )
                
                my_pooled_prompt_embeds = torch.tensor(
                    output_numpy,  
                    dtype=torch.float16,
                    device="cuda:0"
                )
                image = pipe(
                    prompt=None,
                    prompt_embeds=blank_prompt_embeds,
                    pooled_prompt_embeds=my_pooled_prompt_embeds,
                    num_inference_steps=30,
                    guidance_scale=3.0
                ).images[0]

                st.image(image, caption="Generated Image", use_column_width=True)

            except Exception as e:
                st.error(f"An error occurred during image generation: {e}")
            
            finally:
                os.remove(temp_filepath)

if __name__ == "__main__":
    main()