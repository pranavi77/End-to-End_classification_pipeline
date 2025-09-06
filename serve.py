import os, io
from fastapi import FastAPI, File, UploadFile, Request
from pydantic import BaseModel
import onnxruntime as ort, numpy as np
from PIL import Image

MODEL_PATH = os.environ.get("MODEL_PATH", "model_int8_qdq.onnx")
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

app = FastAPI()
LABELS = ["class1","class2"]

class Input(BaseModel):
    x: list  # [N,3,224,224] or [1,3,224,224]

def softmax(z):
    z = np.array(z); z -= z.max(axis=1, keepdims=True)
    return (np.exp(z)/np.exp(z).sum(axis=1, keepdims=True)).tolist()

def preprocess_image(b):
    img = Image.open(io.BytesIO(b)).convert("RGB").resize((224,224))
    x = np.asarray(img, dtype=np.float32)
    x = np.transpose(x, (2,0,1))[None, ...]
    return x

@app.get("/ping")          # SageMaker health check
def ping(): return {"status":"ok"}

@app.post("/invocations")  # SageMaker invoke (JSON only)
async def invocations(req: Request):
    if "application/json" not in req.headers.get("content-type",""):
        return {"error":"use application/json"}, 415
    body = await req.json()
    arr = np.array(body.get("x"), dtype=np.float32)
    if arr.ndim == 3: arr = arr[None]
    y = sess.run(None, {"input": arr})[0]
    probs = softmax(y)
    top1 = {"label": LABELS[int(np.argmax(probs[0]))], "prob": float(max(probs[0]))}
    return {"top1": top1, "probs": probs}

# Optional local routes
@app.get("/health")
def health(): return {"ok": True}

@app.post("/predict")
def predict(inp: Input):
    arr = np.array(inp.x, dtype=np.float32)
    if arr.ndim == 3: arr = arr[None]
    y = sess.run(None, {"input": arr})[0]
    probs = softmax(y); top1={"label":LABELS[int(np.argmax(probs[0]))],"prob":float(max(probs[0]))}
    return {"top1": top1, "probs": probs}

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    arr = preprocess_image(await file.read())
    y = sess.run(None, {"input": arr})[0]
    probs = softmax(y); top1={"label":LABELS[int(np.argmax(probs[0]))],"prob":float(max(probs[0]))}
    return {"top1": top1, "probs": probs}