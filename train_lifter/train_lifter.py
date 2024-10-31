import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Extract 2D Keypoints from Video
def extract_2d_keypoints(video_path, output_file, num_joints=33):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    keypoints_2d = []
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                frame_keypoints = []
                for i in range(num_joints):
                    landmark = results.pose_landmarks.landmark[i]
                    frame_keypoints.append([landmark.x, landmark.y])
                keypoints_2d.append(frame_keypoints)
            else:
                keypoints_2d.append([[np.nan, np.nan]] * num_joints)

    cap.release()
    keypoints_2d = np.array(keypoints_2d, dtype=np.float32)
    np.save(output_file, keypoints_2d)
    print(f"2D keypoints saved to {output_file}")

# Step 2: Generate Synthetic 3D Keypoints
def generate_synthetic_3d_keypoints(input_2d_file, output_3d_file, depth_scale=100):
    keypoints_2d = np.load(input_2d_file)
    num_frames, num_joints, _ = keypoints_2d.shape
    keypoints_3d = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    keypoints_3d[..., :2] = keypoints_2d

    for frame_idx in range(num_frames):
        for joint_idx in range(num_joints):
            if not np.isnan(keypoints_2d[frame_idx, joint_idx, 1]):
                z_value = depth_scale * (1 - keypoints_2d[frame_idx, joint_idx, 1])
                keypoints_3d[frame_idx, joint_idx, 2] = z_value
            else:
                keypoints_3d[frame_idx, joint_idx, 2] = np.nan

    np.save(output_3d_file, keypoints_3d)
    print(f"Synthetic 3D keypoints saved to {output_3d_file}")

# Step 3: Define and Train the 3D Lifting Model
class Simple3DLiftingModel(nn.Module):
    def __init__(self, input_dim=34, output_dim=51, hidden_dim=128):
        super(Simple3DLiftingModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def load_data_2d_3d(data_2d_file, data_3d_file):
    data_2d = np.load(data_2d_file)
    data_3d = np.load(data_3d_file)

    # Use only the 17 relevant joints
    relevant_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    data_2d = data_2d[:, relevant_joints, :].reshape(data_2d.shape[0], -1)
    data_3d = data_3d[:, relevant_joints, :].reshape(data_3d.shape[0], -1)

    return data_2d, data_3d

def train_model(train_loader, model, epochs=20, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

    return model

# Main function to run all steps
def main(video_path, output_model_file):
    # File paths for temporary 2D and 3D keypoint data
    temp_2d_keypoints = "2d_keypoints.npy"
    temp_3d_keypoints = "3d_keypoints.npy"

    # Step 1: Extract 2D keypoints
    extract_2d_keypoints(video_path, temp_2d_keypoints)

    # Step 2: Generate synthetic 3D keypoints
    generate_synthetic_3d_keypoints(temp_2d_keypoints, temp_3d_keypoints)

    # Step 3: Load data and train the model
    data_2d, data_3d = load_data_2d_3d(temp_2d_keypoints, temp_3d_keypoints)
    train_dataset = TensorDataset(torch.tensor(data_2d, dtype=torch.float32),
                                  torch.tensor(data_3d, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = Simple3DLiftingModel(input_dim=34, output_dim=51)
    trained_model = train_model(train_loader, model, epochs=20)
    torch.save(trained_model.state_dict(), output_model_file)
    print(f"Model saved to {output_model_file}")

if __name__ == "__main__":
    main("TaiChi.mp4", "3d_lifting_model.pth")
