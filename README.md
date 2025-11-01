# Kenney Studio 🎨  
_Transform photos into artistic pencil sketches and watercolor paintings using AI._

## Overview
Kenney Studio is an AI art generator that converts any photo into a realistic pencil sketch or watercolor artwork using advanced diffusion models.

## Features
- Upload or capture photos
- Select "Sketch My Photo" to turn photo into an artistic pencil sketch 
- 💡 Fun facts about art techniques
- 🖼️ Interactive gallery with downloadable samples
- ⚙️ Powered by Stable Diffusion + ControlNet

## Tech Stack
- **Streamlit** (UI)
- **Diffusers + ControlNet** (model)
- **HuggingFace Spaces** (deployment)
- **PyTorch**, **OpenCV**, **PIL**

## 🧩 Demo
👉 [Try Kenney Studio on HuggingFace](https://rasheedmrandroid-kenney-studio.hf.space)


## Local Setup
```bash
git clone https://github.com/rasheedmrandroid/Kenney-Studio.git
cd Kenney-Studio
pip install -r requirements.txt
streamlit run app.py