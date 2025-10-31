import streamlit as st
import random, os, io
from PIL import Image, ImageOps
from models.sketch_model import load_sketch_model, generate_sketch

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Kenney Studio 🎨", page_icon="🎨", layout="wide")

# -------------------------------
# STYLING
# -------------------------------
st.markdown("""
<style>
    body { background-color: #f9f9f9; }
    .title { text-align: center; font-size: 2.5rem; color: #2b2b2b; font-weight: 700; margin-bottom: 0.5rem; }
    .subtitle { text-align: center; color: #777; font-size: 1.1rem; margin-bottom: 2rem; }
    .image-container { display: flex; justify-content: center; gap: 2rem; margin-top: 2rem; }
    .image-card { border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); overflow: hidden; }
    .download-btn { display: flex; justify-content: center; margin-top: 0.5rem; }
    .fact-box { background: #fff; border-left: 6px solid #007acc; padding: 1rem; margin: 2rem 0; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# MODEL LOADING (CACHED)
# -------------------------------
@st.cache_resource
def get_model():
    return load_sketch_model()

pipe = get_model()

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<h1 class='title'>Kenney Studio 🎨</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Transform your photo into a realistic pencil sketch masterpiece.</p>", unsafe_allow_html=True)

# -------------------------------
# IMAGE INPUT SECTION (Unified)
# -------------------------------
uploaded_img = None

# st.markdown("### 📸 Choose Your Photo")

option = st.radio("Select source:", ["Upload from device", "Take a live photo"], horizontal=True)

if option == "Upload from device":
    uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        uploaded_img = Image.open(uploaded_file).convert("RGB")

elif option == "Take a live photo":
    camera_input = st.camera_input("Take a photo")
    if camera_input:
        uploaded_img_live = Image.open(camera_input).convert("RGB")

# Display the chosen image (if available)
if uploaded_img is not None:
    st.image(uploaded_img, caption="Selected Image", use_container_width=True)
    st.markdown("<div style='text-align:center; margin-top:20px;'>", unsafe_allow_html=True)
    # process_btn = st.button("✨ Sketch My Photo", type="primary")
    # st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# PROCESS BUTTON
# -------------------------------
if uploaded_img is not None:
    st.markdown("<div style='text-align:center; margin-top:20px;'>", unsafe_allow_html=True)
    process_btn = st.button("✨ Sketch My Photo", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if process_btn:
        with st.spinner("🖋️ Drawing your sketch... please wait."):
            # Fix orientation + resize before sending to model
            uploaded_img = ImageOps.exif_transpose(uploaded_img)
            uploaded_img = uploaded_img.resize((512, 512))

            sketch_result = generate_sketch(pipe, uploaded_img)
        
        # Convert for download
        buf = io.BytesIO()
        sketch_result.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        # Display results side by side
        colA, colB = st.columns(2)
        with colA:
            st.image(uploaded_img, caption="Original Image", use_container_width=True)
        with colB:
            st.image(sketch_result, caption="Pencil Sketch", use_container_width=True)
            st.download_button("⬇️ Download Sketch", data=byte_im, file_name="kenney_sketch.jpg", mime="image/jpeg")

# -------------------------------
# GALLERY SECTION
# -------------------------------
st.markdown("### 🎨 Pencil Sketch Gallery")
gallery_path = "assets/gallery"
facts_path = "assets/facts.json"

# Load fun facts (if available)
facts = []
if os.path.exists(facts_path):
    import json
    with open(facts_path, "r") as f:
        facts = json.load(f)

if os.path.exists(gallery_path):
    all_imgs = [f for f in os.listdir(gallery_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    show_imgs = random.sample(all_imgs, min(5, len(all_imgs)))
    cols = st.columns(5)

    for i, img_name in enumerate(show_imgs):
        with cols[i]:
            img_path = os.path.join(gallery_path, img_name)
            img = Image.open(img_path)
            st.image(img, use_container_width=True)

            # Fun fact under each image
            if facts:
                fact = random.choice(facts)
                st.markdown(
                    f"<div class='fact-box' style='font-size:0.9rem; margin:0.5rem 0 1rem 0;'>💡 <i>{fact}</i></div>",
                    unsafe_allow_html=True
                )

            # Download button below fact
            with open(img_path, "rb") as f:
                st.download_button("⬇️ Download", data=f, file_name=img_name)
else:
    st.info("Gallery not found. Please add images to `assets/gallery/`.")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("<hr><p style='text-align:center; color:#888;'>© 2025 Kenney Studio | Crafted with ❤️ by Mr. Android</p>", unsafe_allow_html=True)