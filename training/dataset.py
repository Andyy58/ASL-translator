from numpy.random import shuffle
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d


class ASLDataset(Dataset):
    def __init__(self, data_dir, max_frames=100):
        """
        data_dir: path to 'train' or 'val' folder
        max_frames: Fixed length to pad/truncate videos to
        """

        self.data_dir = data_dir
        self.max_frames = max_frames
        self.classes = sorted([cls for cls in os.listdir(data_dir) if not cls.startswith("._")])
        self.class_to_idx = {word: i for i, word in enumerate(self.classes)}
        self.file_paths = []
        self.labels = []

        for word in self.classes:
            word_dir = os.path.join(data_dir, word)
            if not os.path.isdir(word_dir):
                continue

            for file_name in os.listdir(word_dir):
                if file_name.endswith(".npy") and not file_name.startswith("._"):
                    self.file_paths.append(os.path.join(word_dir, file_name))
                    self.labels.append(self.class_to_idx[word])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        video_data = np.load(self.file_paths[idx])
        frame_count = video_data.shape[0]

        x = np.linspace(0, 1, num=frame_count)
        x_new = np.linspace(0, 1, num=self.max_frames)

        f = interp1d(x, video_data, axis=0, kind='linear', fill_value="extrapolate")
        interpolated_data = f(x_new).astype(np.float32)

        return torch.tensor(interpolated_data), torch.tensor(self.labels[idx], dtype=torch.long)

if __name__ == "__main__":
    train_dir = os.path.join(os.getcwd(), "..", "feature_extraction", "wlasl100_features", "train")

    train_dataset = ASLDataset(data_dir=train_dir, max_frames=100)
    print(f"Total training videos found: {len(train_dataset)}")
    print(f"Total classes (words): {len(train_dataset.classes)}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    for features, labels in train_loader:
        print(f"\nBatch Features Shape: {features.shape}")
        print(f"Batch Labels Shape: {labels.shape}")
        print(f"First 5 labels in this batch: {labels[:5]}")
        break