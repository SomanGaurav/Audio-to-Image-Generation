import os
import tempfile
import numpy as np
import librosa
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from torchvggish import vggish, vggish_input
import streamlit as st
import laion_clap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- MODEL CLASSES ----------------------
class TransformationNetwork(nn.Module):
    def __init__(self, input_dim=512, output_dim=2048):
        super(TransformationNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),

            nn.Linear(512, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),

            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.4),

            nn.Linear(1024, output_dim)
        )
    def forward(self, x):
        return self.layers(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiOutputTransformer(nn.Module):
    def __init__(self, input_dim=128, d_model=512, nhead=8, num_layers=4, ff_dim=1024):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.upsample = nn.Linear(5, 20)
        self.out1 = nn.Linear(d_model, 768)
        self.out2 = nn.Linear(d_model, 1280)
        self.out3 = nn.Linear(d_model, 4096)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)

        x = x.transpose(1, 2)
        x = self.upsample(x)
        x = x.transpose(1, 2)

        return self.out1(x), self.out2(x), self.out3(x)

# ---------------------- CACHED LOADERS ----------------------
@st.cache_resource
def load_pipeline():
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16
    ).to("cuda:1")
    return pipe

@st.cache_resource
def load_vggish():
    return vggish()

@st.cache_resource
def load_clap():
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    return model.to(device)

@st.cache_resource
def load_transformers():
    pooled_mapper = TransformationNetwork().to(device)
    pooled_mapper.load_state_dict(torch.load("best_clap_mapper.pth", map_location=device))
    pooled_mapper.eval()

    transformer = MultiOutputTransformer().to(device)
    transformer.load_state_dict(torch.load("best_model_two_stage.pt", map_location=device))
    transformer.eval()

    return pooled_mapper, transformer

# ---------------------- AUDIO ENCODING ----------------------
def compute_vggish(file_path, model):
    audio, sr = librosa.load(file_path, sr=16000, mono=True)
    example = vggish_input.waveform_to_examples(audio, sr)
    example = torch.tensor(example).float()
    with torch.no_grad():
        emb = model(example).cpu().numpy()
    if emb.shape[0] < 5:
        emb = np.vstack([emb, np.zeros((5 - emb.shape[0], emb.shape[1]))])
    else:
        emb = emb[:5]
    return torch.tensor(emb).float().unsqueeze(0)

# ---------------------- STREAMLIT UI ----------------------
def main():
    st.title("🎵 Audio → Stable Diffusion 3 Image Generator")

    st.info("Loading models... please wait ⏳ (only happens once)")
    with st.spinner("Initializing all models..."):
        pipe = load_pipeline()
        vggish_model = load_vggish()
        clap_model = load_clap()
        pooled_mapper, transformer = load_transformers()
    st.success("Models loaded successfully. Upload an audio file to begin.")

    uploaded_audio = st.file_uploader("Upload .wav audio", type=["wav"])

    if uploaded_audio:
        st.audio(uploaded_audio)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_audio.read())
            audio_path = tmp.name

        with st.spinner("Extracting audio embeddings..."):
            vggish_emb = compute_vggish(audio_path, vggish_model)

            with torch.no_grad():
                clap_emb = clap_model.get_audio_embedding_from_filelist([audio_path], use_tensor=True)

            sd3_pool = pooled_mapper(clap_emb)

            y1, y2, y3 = transformer(vggish_emb.to(device))

            y1 = F.pad(y1, (0,0,0,77-y1.size(1)))
            y2 = F.pad(y2, (0,0,0,77-y2.size(1)))
            y3 = F.pad(y3, (0,0,0,256-y3.size(1)))

            clip = torch.cat([y1, y2], dim=-1)
            clip = F.pad(clip, (0, y3.shape[-1] - clip.shape[-1]))
            prompt_embeds = torch.cat([clip, y3], dim=-2)

        with st.spinner("Generating image using Stable Diffusion 3..."):
            prompt_embeds = prompt_embeds.to("cuda:1", dtype=torch.float16)
            sd3_pool = sd3_pool.to("cuda:1", dtype=torch.float16)

            img = pipe(
                prompt=None,
                negative_prompt=None,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=sd3_pool,
                num_inference_steps=28,
                guidance_scale=7.0
            ).images[0]


        st.image(img, caption="Generated Image")

if __name__ == "__main__":
    main()
