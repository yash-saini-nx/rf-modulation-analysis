"""Script to generate and save confusion matrices for trained models."""
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix
from src.dataset import prepare_dataset
from src.model import build_model
import torch.utils.data as data

ROOT = Path(__file__).resolve().parent

def main():
    print("Loading data...")
    dataset = prepare_dataset("data/signals.mat")
    test_tensor = torch.utils.data.TensorDataset(
        torch.tensor(dataset.X_test, dtype=torch.float32),
        torch.tensor(dataset.y_test, dtype=torch.long)
    )
    test_loader = torch.utils.data.DataLoader(test_tensor, batch_size=512, shuffle=False)
    
    with open("models/metadata.json", "r") as f:
        metadata = json.load(f)
    classes = metadata["classes"]
    num_classes = len(classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for model_name in ["cnn_baseline", "cldnn_lite"]:
        model_path = ROOT / "models" / f"{model_name}.pt"
        if not model_path.exists():
            continue
            
        print(f"Evaluating {model_name}...")
        model = build_model(model_name, num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for iq, target in test_loader:
                iq = iq.to(device)
                logits = model(iq)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(target.numpy())
                
        cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))
        
        # Save as CSV for the dashboard
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        out_path = ROOT / "models" / f"{model_name}_confusion_matrix.csv"
        cm_df.to_csv(out_path)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
