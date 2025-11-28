# Audio to Image Generation using Baseline Diffusion Model

### IE643 Course Project — Deep Learning: Theory and Practice  
### Team: **The ReLUtionaries**

**Authors:**  
- Gaurav Soman (25M1516) — 25m1516@iitb.ac.in  
- Siddhesh Madkaikar (25M1526) — 25m1526@iitb.ac.in  

---

## 📌 Abstract
We propose an audio-to-image generation framework using **Stable Diffusion 3-Medium**, keeping its diffusion backbone completely frozen. A lightweight adapter (V2D) maps **VGGish audio embeddings** to the pooled text-embedding space of SD-3, enabling semantic image generation directly from real-world audio.  
Results demonstrate that lightweight adapter networks can repurpose text-to-image models for audio-conditioned generation without modifying the diffusion model.

---

## 🔥 Key Contributions
- **V2D Mapper:** Maps VGGish embeddings (128-D) to SD3 pooled embedding space (2048-D).
- **Frozen Diffusion Backbone:** No finetuning or modifications to Stable Diffusion 3 model.
- **Direct Audio → Semantic Embedding Mapping** without tokenization.
- **Curated Dataset Pipeline** based on AudioSet with refined embedding pairs.
- **End-to-End real-time system** with GUI for unseen audio inference.

---

## 🧠 Motivation
- Text prompts are not always naturally aligned with visual content.
- Audio represents real-world events and holds inherent semantic alignment with visuals.
- Prior approaches rely on intermediate caption-generation, limiting flexibility.
- Our method directly conditions Stable Diffusion on audio without text conversion.

---

## 📍 System Workflow
- to be added
---

## 🛠 Methodology
- A lightweight Mapper Network learns a transformation from **128-D VGGish** features to **2048-D SD3 pooled text embeddings**.
- Works inside pooled semantic space for efficiency and real-time inference.
- Enables direct multimodal transfer learning without modifying SD3 internals.

---

## 📂 Dataset Details
- **Dataset:** AudioSet (balanced subset with ≥ 60% class quality)
- **Data size:** ~34.5 GB of 10-second audio clips
- Created meaningful visual prompts (e.g. *“an image of a dog”*) and extracted SD-3 embeddings.
- Built **36k initial samples**, refined to **~25k specific pairs**.
- Additional datasets: **AudioCaps, ESC50** for novelty evaluation.

---

## 🧪 Novelty Assessment
- Partial-token strategy: first **20 pooled tokens + zero padding**
- Only early token space contained meaningful semantic information.
- Validated cross-modal alignment via AudioCaps & ESC50 experiments.

---

## 📊 Results

| Methods | Combined Loss | MSE | Cosine Similarity |
|---------|--------------|-----|-------------------|
| Intensive Assessment | 0.225 | 0.31 | 0.81 |
| Novelty Experiment 1 | 0.9863 | 0.5342 | 0.5479 |
| Novelty Experiment 2 | 0.4360 | 0.5462 | 0.5640 |
| Novelty Experiment 3 | 0.2311 | 0.9653 | 0.7689 |

Qualitative image examples include birds, dogs, and marine animals produced directly from sound inputs.

---

## 🏁 Conclusion
- Demonstrated **audio-to-image** generation with Stable Diffusion 3 kept frozen.
- Introduced **token-partial embedding mapping** and audio-semantic alignment.
- Highlights audio as a strong conditioning modality for future multimodal diffusion research.

---

## 📚 References
1. Yariv et al., *AudioToken: Adaptation of Text-Conditioned Diffusion Models for Audio-to-Image Generation*, INTERSPEECH 2023.  
2. Sung-Bin Kim et al., *Sound to Visual Scene Generation by Audio-to-Visual Latent Alignment*, CVPR 2023.

---

## 🙏 Acknowledgments
We thank **IIT Bombay** and **IE643 instructors** for guidance and computational support.  
We acknowledge **AudioSet, AudioCaps, ESC50**, and **Stable Diffusion 3** models.

---

## 🔗 Project Links
| Resource | Link |
|---------|------|
| GitHub Repository | *QR provided in project poster* |
| Demo Video | *QR provided in project poster* |

---

## 🚀 Future Work
- Support generative transformers and video diffusion models.
- Add multi-prompt layered audio conditioning.
- Integrate audio localization for spatial control.

---

