"""
File: backend/api/index.py
Description: FastAPI backend for MRI Brain Tumor classification (Vercel deployment)
"""

import os, io, uuid, base64
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

# ---------------- CONFIG ---------------- #
HF_REPO_ID = "your-username/detectra-model"  # Change this!
MODEL_FILENAME = "best.pt"
MODEL_VERSION = "v1.0"
ALLOWED_ORIGINS = ["*"]  # Change to your frontend domain

# ---------------- MODEL ---------------- #
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
CLASS_NAMES = None

def load_model():
    global model, CLASS_NAMES
    
    if model is not None:
        return model, CLASS_NAMES
    
    print(f"Loading model from Hugging Face: {HF_REPO_ID}")
    
    # Download from Hugging Face
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
        cache_dir="/tmp/hf_cache"
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    CLASS_NAMES = checkpoint.get("class_names", ["glioma","meningioma","notumor","pituitary"])
    
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)
    
    print(f"Model loaded successfully with {len(CLASS_NAMES)} classes")
    return model, CLASS_NAMES

# ---------------- PREPROCESSING ---------------- #
def preprocess_image(image: Image.Image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# ---------------- GRAD-CAM ---------------- #
class GradCAM:
    def __init__(self, model, target_layer="layer4"):
        self.model = model
        self.gradients = None
        self.activations = None
        layer = dict(model.named_modules())[target_layer]
        layer.register_forward_hook(self._forward_hook)
        layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        out = self.model(input_tensor)
        score = out[0, class_idx]
        score.backward()
        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224,224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam

def overlay_heatmap(image: Image.Image, mask, alpha=0.5):
    import numpy as np
    mask = (mask * 255).astype("uint8")
    mask_img = Image.fromarray(mask).resize(image.size)
    mask_img = mask_img.convert("RGB")
    blended = Image.blend(image.convert("RGB"), mask_img, alpha)
    buf = io.BytesIO()
    blended.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ---------------- FASTAPI APP ---------------- #
app = FastAPI(title="Brain Tumor Classification API", version=MODEL_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- SCHEMAS ---------------- #
class Feedback(BaseModel):
    request_id: str
    correct_label: str
    comment: str = ""

# ---------------- ROUTES ---------------- #
@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "model_version": MODEL_VERSION,
        "service": "Brain Tumor Classification API"
    }

@app.get("/model-info")
def model_info():
    # Load model to get class names
    _, class_names = load_model()
    return {
        "model_version": MODEL_VERSION,
        "classes": class_names
    }

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...), explain: bool = Form(False)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image")
    
    # Load model (lazy loading)
    global model, CLASS_NAMES
    model, CLASS_NAMES = load_model()
    
    request_id = str(uuid.uuid4())

    img = Image.open(io.BytesIO(await file.read()))
    tensor = preprocess_image(img)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
        top_idx = int(probs.argmax())
        prediction = CLASS_NAMES[top_idx]

    response = {
        "request_id": request_id,
        "prediction": prediction,
        "probabilities": {CLASS_NAMES[i]: float(p) for i,p in enumerate(probs)}
    }

    if explain:
        gradcam_obj = GradCAM(model)
        mask = gradcam_obj.generate(tensor, top_idx)
        heatmap_b64 = overlay_heatmap(img, mask)
        response["gradcam"] = heatmap_b64

    return response

@app.post("/feedback")
def submit_feedback(f: Feedback):
    # Note: SQLite won't persist on Vercel serverless
    # Consider using external database (Supabase, MongoDB Atlas)
    # For now, just acknowledge receipt
    print(f"Feedback received: {f.dict()}")
    return {
        "status": "feedback received",
        "note": "Feedback logging disabled in serverless deployment"
    }

# For Vercel (important!)
app = app