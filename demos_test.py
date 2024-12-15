import os
import torch
from torch.utils.data import DataLoader
import pandas as pd

# Import custom modules
from models_demos import slowfast_r50_Modelim, r3d_18_Modelim, x3d_Modelim_medium  # Import models
from dataset_demos import TestVideoDataset  # Import dataset class

# Check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Constants
CLASS_NAMES = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness"]
BATCH_SIZE = 4
FRAME_SIZE = (224, 224)
NUM_FRAMES = 32

# Model paths
MODEL_PATHS = {
    "slowfast": r"model_folder\slowfast_r50_DEMOS.pth",
    "r3d_18": r"model_folder\r3d_18_Model_DEMOS.pth",
    "x3d_medium": r"model_folder\x3d_Medium_DEMOS.pth"
}

# Define dataset root directory
DATASET_DIR = r"test_files"

# Output Excel file
OUTPUT_EXCEL = "evaluation_results.xlsx"


def load_model(model_class, model_path, num_classes):
    """
    Load a pre-trained model and initialize it with weights from a file.
    Attempts to load the full model first, then state_dict, and finally checkpoint (_check.pth).

    Args:
        model_class (nn.Module): The model class to instantiate.
        model_path (str): Path to the model weights file.
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: Loaded and initialized model or None if loading fails.
    """
    try:
        # Try loading as a full model
        model = torch.load(model_path, map_location=device)
        print(f"Successfully loaded full model from {model_path}")
    except Exception as e_full:
        print(f"Failed to load full model: {e_full}. Attempting to load state_dict.")
        try:
            # Try loading as a state_dict
            model = model_class(num_classes=num_classes).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Successfully loaded state_dict from {model_path}")
        except Exception as e_state:
            print(f"Failed to load state_dict: {e_state}. Checking for checkpoint file.")
            checkpoint_path = model_path.replace(".pth", "_check.pth")
            try:
                # Try loading as a checkpoint
                model = model_class(num_classes=num_classes).to(device)
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                print(f"Successfully loaded checkpoint from {checkpoint_path}")
            except Exception as e_checkpoint:
                print(f"Failed to load checkpoint: {e_checkpoint}. Skipping model.")
                model = None
    if model:
        model.eval()
    return model


def evaluate_videos(test_loader, model, model_type, class_names):
    """
    Evaluate videos using the specified model and return predictions.

    Args:
        test_loader (DataLoader): DataLoader for the test dataset.
        model (nn.Module): Loaded model for evaluation.
        model_type (str): Type of the model ('slowfast' or 'single-tensor').
        class_names (list): List of class names.

    Returns:
        dict: Evaluation results with video paths and predictions.
    """
    results = {}
    if model is None:
        print(f"Skipping evaluation for model. It could not be loaded.")
        return results

    with torch.no_grad():
        for inputs, video_paths in test_loader:
            if model_type == "slowfast":
                inputs = [i.to(device) for i in inputs]  # Slow and fast pathways
                print(f"Slow Pathway Shape: {inputs[0].shape}, Fast Pathway Shape: {inputs[1].shape}")
            elif model_type == "single-tensor":
                inputs = inputs[1].to(device)  # Fast pathway (single tensor)
                print(f"Input Shape: {inputs.shape}")

            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predictions = probabilities.argmax(dim=1).cpu().numpy()

            for i, video_path in enumerate(video_paths):
                results[video_path] = {
                    "Predicted Label": class_names[predictions[i]],
                    "Probabilities": probabilities[i].cpu().numpy().tolist()
                }
                print(f"Processed {video_path}: Predicted {class_names[predictions[i]]}")

    return results


def save_results_to_excel(results, output_path):
    """
    Save evaluation results to an Excel file.

    Args:
        results (dict): Dictionary with evaluation results.
        output_path (str): Path to save the Excel file.
    """
    with pd.ExcelWriter(output_path) as writer:
        for model_name, model_results in results.items():
            data = []
            for video_path, result in model_results.items():
                row = {
                    "Video Path": video_path,
                    "Predicted Label": result["Predicted Label"],
                }
                row.update({f"Prob_{i}": prob for i, prob in enumerate(result["Probabilities"])})
                data.append(row)
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=model_name, index=False)
    print(f"Results saved to {output_path}")


# Initialize dataset and dataloader
test_dataset = TestVideoDataset(
    root_dir=DATASET_DIR,
    frame_size=FRAME_SIZE,
    num_frames=NUM_FRAMES
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load models
models = {
    "SlowFast": load_model(slowfast_r50_Modelim, MODEL_PATHS["slowfast"], len(CLASS_NAMES)),
    "R3D_18": load_model(r3d_18_Modelim, MODEL_PATHS["r3d_18"], len(CLASS_NAMES)),
    "X3D_Medium": load_model(x3d_Modelim_medium, MODEL_PATHS["x3d_medium"], len(CLASS_NAMES))
}

# Evaluate models
all_results = {}
for model_name, model in models.items():
    print(f"\nEvaluating {model_name} model:")
    model_type = "slowfast" if model_name == "SlowFast" else "single-tensor"
    all_results[model_name] = evaluate_videos(test_loader, model, model_type, CLASS_NAMES)

# Save results to Excel
save_results_to_excel(all_results, OUTPUT_EXCEL)
