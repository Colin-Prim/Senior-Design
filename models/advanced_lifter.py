import torch
import torch.nn as nn
import numpy as np

class AdvancedLifter:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model()
        self._load_weights(model_path)

    def _build_model(self):
        # Adjusted input and output dimensions
        model = nn.Sequential(
            nn.Linear(66, 128),  # Adjusted input size to match 66 input features
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 99)   # Output size to match 33 3D keypoints
        )
        return model.to(self.device)

    def _load_weights(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Check for unexpected keys and load only matching keys
            model_dict = self.model.state_dict()
            matched_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            self.model.load_state_dict(matched_state_dict, strict=False)

            print(f"Model weights loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model weights: {e}")

    def lift(self, keypoints_2d):
        try:
            keypoints_2d_flat = keypoints_2d.flatten().reshape(1, -1)
            input_tensor = torch.tensor(keypoints_2d_flat, dtype=torch.float32).to(self.device)
            output_tensor = self.model(input_tensor)

            keypoints_3d = output_tensor.detach().cpu().numpy().reshape(-1, 3)
            print(f"Converted 3D Keypoints: {keypoints_3d}")  # Debugging output
            return keypoints_3d
        except Exception as e:
            print(f"Error during 3D lifting: {e}")
            return None

