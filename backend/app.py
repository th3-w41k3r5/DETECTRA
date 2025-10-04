"""
File: backend/app.py
Description: FastAPI backend for MRI Brain Tumor classification
Dependencies: fastapi, uvicorn, torch, torchvision, pillow, sqlalchemy
Run: uvicorn backend.app:app --reload --port 8000
"""

import os, io, uuid, json, base64, sqlite3, datetime
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------- CONFIG ---------------- #
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
DB_PATH = os.getenv("DB_PATH", "backend/feedback.db")
MODEL_VERSION = "v1.0"
ALLOWED_ORIGINS = ["*"]  # change to your frontend domain (e.g. v0.dev domain)

# ---------------- DATABASE ---------------- #
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT,
            correct_label TEXT,
            comment TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_feedback(request_id: str, correct_label: str, comment: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO feedback (request_id, correct_label, comment, created_at) VALUES (?,?,?,?)",
                 (request_id, correct_label, comment, datetime.datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

init_db()

# ---------------- MODEL ---------------- #
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    print(f"Loading model from {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint.get("class_names", ["glioma","meningioma","notumor","pituitary"])
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)
    return model, class_names

model, CLASS_NAMES = load_model()

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

# ---------------- GRAD-CAM (optional explainability) ---------------- #
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

gradcam = GradCAM(model)

def overlay_heatmap(image: Image.Image, mask, alpha=0.5):
    import numpy as np
    from PIL import ImageEnhance
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
@app.get("/health")
def health():
    return {"status":"ok","device":device,"model_version":MODEL_VERSION,"num_classes":len(CLASS_NAMES)}

@app.get("/model-info")
def model_info():
    return {"model_version":MODEL_VERSION,"classes":CLASS_NAMES}

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...), explain: bool = Form(False)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image")
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
        mask = gradcam.generate(tensor, top_idx)
        heatmap_b64 = overlay_heatmap(img, mask)
        response["gradcam"] = heatmap_b64

    return response

@app.post("/feedback")
def submit_feedback(f: Feedback):
    save_feedback(f.request_id, f.correct_label, f.comment)
    return {"status":"feedback saved"}

# ---------------- RUN ---------------- #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
