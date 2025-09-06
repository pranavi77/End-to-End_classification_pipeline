# train_mlflow.py  (final, simple, stable: FP32 train + ONNX export + ONNX INT8 quant)
import os, numpy as np, time
import torch, torch.nn as nn, torch.optim as optim, torch.onnx
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import get_model
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import mlflow, mlflow.pytorch
from mlflow.models import infer_signature
from onnxruntime.quantization import quantize_dynamic, QuantType

# force logs to project root
MLRUNS_DIR = os.path.abspath("./mlruns")
mlflow.set_tracking_uri("file:" + MLRUNS_DIR)
mlflow.set_experiment("classification-exp")

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # paths (Hydra-safe)
    data_root  = to_absolute_path(cfg.data_dir)
    model_root = to_absolute_path(cfg.model_dir)
    os.makedirs(model_root, exist_ok=True)

    # data
    tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    train_ds = datasets.ImageFolder(root=os.path.join(data_root, "train"), transform=tfm)
    val_ds   = datasets.ImageFolder(root=os.path.join(data_root, "val"),   transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size)

    # model / loss / opt
    model = get_model(cfg.num_classes).to(device)
    crit  = nn.CrossEntropyLoss()
    opt   = optim.Adam(model.parameters(), lr=cfg.lr)

    with mlflow.start_run():
        mlflow.log_params({"epochs": cfg.epochs, "lr": cfg.lr, "batch_size": cfg.batch_size})

        # train
        for e in range(cfg.epochs):
            model.train(); loss_sum = 0.0
            for X,y in train_loader:
                X,y = X.to(device), y.to(device)
                opt.zero_grad()
                out  = model(X)
                loss = crit(out,y)
                loss.backward(); opt.step()
                loss_sum += loss.item()
            mlflow.log_metric("train_loss", loss_sum/len(train_loader), step=e)
            print(f"Epoch {e+1}, Loss: {loss_sum/len(train_loader):.4f}")

        # val
        model.eval(); correct=total=0
        with torch.no_grad():
            for X,y in val_loader:
                X,y = X.to(device), y.to(device)
                pred = model(X).argmax(1)
                correct += (pred==y).sum().item(); total += y.size(0)
        acc = 100*correct/total if total else 0.0
        mlflow.log_metric("val_accuracy", acc)
        print(f"Val Accuracy: {acc:.2f}%")

        # save FP32 .pth
        pth_path = os.path.join(model_root, "model.pth")
        torch.save(model.state_dict(), pth_path)
        mlflow.log_artifact(pth_path)

        # log model w/ signature
        example_np = np.random.randn(1,3,224,224).astype(np.float32)
        with torch.no_grad():
            out_np = model(torch.from_numpy(example_np).to(device)).cpu().numpy()
        sig = infer_signature(example_np, out_np)
        mlflow.pytorch.log_model(model, artifact_path="pytorch-model", input_example=example_np, signature=sig)

        # --- ONNX FP32 export ---
        model_cpu = model.to("cpu").eval()
        dummy = torch.randn(1,3,224,224)
        onnx_fp32 = os.path.join(model_root, "model_fp32.onnx")
        torch.onnx.export(model_cpu, dummy, onnx_fp32,
                          input_names=["input"], output_names=["output"], opset_version=12)
        mlflow.log_artifact(onnx_fp32)

        # --- ONNX INT8 quantization (no deprecated PyTorch APIs) ---
        onnx_int8 = os.path.join(model_root, "model_int8.onnx")
        quantize_dynamic(onnx_fp32, onnx_int8, weight_type=QuantType.QInt8)
        mlflow.log_artifact(onnx_int8)

        # sizes
        fp32_mb = os.path.getsize(onnx_fp32)/1e6
        int8_mb = os.path.getsize(onnx_int8)/1e6
        print(f"ONNX size FP32: {fp32_mb:.2f} MB | INT8: {int8_mb:.2f} MB")
        mlflow.log_metric("onnx_size_mb_fp32", fp32_mb)
        mlflow.log_metric("onnx_size_mb_int8", int8_mb)

if __name__ == "__main__":
    main()