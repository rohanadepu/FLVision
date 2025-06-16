import os
import sys
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    sys.path.append("..")
    from NIDS import NIDSModule
    from dataset_iotbotnet import loadIOTBOTNET
except Exception as e:
    raise ImportError("Unable to import model or dataset loader.") from e

def evaluate_model(model, test_paths, feature_cols):
    model.eval()  # Set model to evaluation mode
    y_true = []
    y_pred = []

    with torch.no_grad():
        for path in test_paths:
            df = pd.read_feather(path)
            features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
            labels = df['Label'].apply(lambda x: 0 if x == 'Normal' else 1).values

            outputs = model(features)
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_true.extend(labels)
            y_pred.extend(preds)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n>>> Model Evaluation Results <<<")
    print(f"Test Accuracy :  {accuracy:.4f}")
    print(f"Test Precision:  {precision:.4f}")
    print(f"Test Recall   :  {recall:.4f}")
    print(f"Test F1 Score :  {f1:.4f}")
    print(">>> Evaluation complete.")

def main():
    # Load model
    model_path = "out/nids_final_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Train the model first.")

    model = NIDSModule()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print(f"Loaded model from {model_path}")

    # Load test dataset paths
    _, test_paths, feature_names = loadIOTBOTNET()
    print(f"Found {len(test_paths)} test chunks.")

    # Evaluate
    evaluate_model(model, test_paths, feature_names)

if __name__ == "__main__":
    main()
