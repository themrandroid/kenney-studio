from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from huggingface_hub import login
import torch, cv2, os, numpy as np
from PIL import Image

# -----------------------------
# 🧠 Configure local cache paths
# -----------------------------
CACHE_DIR = "/app/.cache/huggingface"

os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

# -----------------------------
# 🔐 Login to Hugging Face
# -----------------------------
token = os.environ.get("HF_TOKEN")
if token:
    try:
        login(token=token)
    except Exception as e:
        print(f"[Warning] Failed to log in to Hugging Face Hub: {e}")
else:
    print("[Info] No HF_TOKEN found. Proceeding without authentication.")

# -----------------------------
# 🎨 Helper functions
# -----------------------------
def dim_background(image, dim_strength=0.5, blur_strength=45):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    mask = np.zeros_like(gray, dtype=np.uint8)
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        x, y = max(0, x - int(w*0.4)), max(0, y - int(h*0.6))
        w, h = int(w*1.8), int(h*2.2)
        cv2.ellipse(mask, (x + w//2, y + h//2), (w//2, h//2), 0, 0, 360, 255, -1)
    else:
        center = (image.shape[1]//2, image.shape[0]//2)
        radius = int(min(image.shape[:2]) * 0.4)
        cv2.circle(mask, center, radius, 255, -1)

    blurred = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    dimmed = (blurred * dim_strength).astype(np.uint8)
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) / 255.0
    result = (image * mask_3d + dimmed * (1 - mask_3d)).astype(np.uint8)
    return result


def apply_vignette(image, intensity=0.6):
    rows, cols = image.shape[:2]
    X_resultant_kernel = cv2.getGaussianKernel(cols, cols*intensity)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, rows*intensity)
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()
    vignette = np.copy(image)
    for i in range(3):
        vignette[:, :, i] = vignette[:, :, i] * mask
    return vignette


# -----------------------------
# ⚙️ Model loader
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_sketch_model():
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        cache_dir=CACHE_DIR
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        cache_dir=CACHE_DIR
    ).to(device)
    return pipe


# -----------------------------
# 🧑‍🎨 Sketch generator
# -----------------------------
def generate_sketch(pipe, input_image, low_threshold=80, high_threshold=160, steps=15):
    import torch
    from PIL import ImageOps

    # Ensure step value is valid (avoid index 1001 errors)
    steps = min(max(steps, 1), 999)

    # Preprocess the image
    image = np.array(input_image)
    image = dim_background(image, dim_strength=0.5, blur_strength=45)
    image = apply_vignette(image)

    # Detect edges
    edges = cv2.Canny(image, low_threshold, high_threshold)
    edges = Image.fromarray(edges)

    prompt = "realistic pencil sketch of a person, graphite texture, clean background, artistic shading"

    # Reset scheduler timesteps each time to prevent index overflow
    pipe.scheduler.set_timesteps(steps, device=pipe.device)

    with torch.no_grad():
        result = pipe(
            prompt,
            image=edges,
            num_inference_steps=steps
        ).images[0]

    return result