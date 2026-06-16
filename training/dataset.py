import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ASLFeatureDataset(Dataset):
    def __init__(
        self,
        root_dir,
        max_frames=100,
        trim_zero_frames=False,
        normalize_hands=False,
        sample_strategy="head",
        zero_frame_threshold=1e-6,
        augment=False,
        noise_std=0.0,
        scale_range=0.0,
        hand_drop_prob=0.0,
        frame_drop_prob=0.0,
    ):
        self.root_dir = root_dir
        self.max_frames = max_frames
        self.trim_zero_frames = trim_zero_frames
        self.normalize_hands = normalize_hands
        self.sample_strategy = sample_strategy
        self.zero_frame_threshold = zero_frame_threshold
        self.augment = augment
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.hand_drop_prob = hand_drop_prob
        self.frame_drop_prob = frame_drop_prob

        if sample_strategy not in {"head", "uniform"}:
            raise ValueError("sample_strategy must be either 'head' or 'uniform'")

        # Translate labels to idx
        self.classes = sorted(
            cls for cls in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, cls))
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.filepaths = []
        self.labels = []

        for word in self.classes:
            word_dir = os.path.join(root_dir, word)
            if os.path.isdir(word_dir):
                for file in os.listdir(word_dir):
                    if file.endswith(".npy"):
                        self.filepaths.append(os.path.join(word_dir, file))
                        self.labels.append(self.class_to_idx[word])

    def __len__(self):
        return len(self.filepaths)

    def _trim_zero_frames(self, matrix):
        frame_has_signal = np.abs(matrix).sum(axis=1) > self.zero_frame_threshold
        if not frame_has_signal.any():
            return matrix[:1]
        return matrix[frame_has_signal]

    @staticmethod
    def _normalize_single_hand(hand):
        # MediaPipe returns zeros when a hand is missing; keep that signal intact.
        if np.abs(hand).sum() == 0:
            return hand

        landmarks = hand.reshape(21, 3)
        wrist = landmarks[0].copy()
        centered = landmarks - wrist

        xy_scale = np.linalg.norm(centered[:, :2], axis=1).max()
        if xy_scale < 1e-6:
            return centered.reshape(-1)
        return (centered / xy_scale).reshape(-1)

    def _normalize_hands(self, matrix):
        normalized = matrix.copy()
        for frame_idx in range(normalized.shape[0]):
            normalized[frame_idx, :63] = self._normalize_single_hand(
                normalized[frame_idx, :63]
            )
            normalized[frame_idx, 63:] = self._normalize_single_hand(
                normalized[frame_idx, 63:]
            )
        return normalized

    def _select_frames(self, matrix):
        total_frames = matrix.shape[0]
        usable_frames = min(total_frames, self.max_frames)

        if total_frames <= self.max_frames:
            return matrix[:usable_frames], usable_frames

        if self.sample_strategy == "uniform":
            indices = np.linspace(0, total_frames - 1, self.max_frames).round().astype(int)
            return matrix[indices], usable_frames

        return matrix[:usable_frames], usable_frames

    def _augment_sequence(self, matrix):
        augmented = matrix.copy()

        if self.scale_range > 0:
            scale = np.random.uniform(1.0 - self.scale_range, 1.0 + self.scale_range)
            nonzero = np.abs(augmented).sum(axis=1, keepdims=True) > self.zero_frame_threshold
            augmented = np.where(nonzero, augmented * scale, augmented)

        if self.noise_std > 0:
            noise = np.random.normal(0.0, self.noise_std, size=augmented.shape).astype(
                np.float32
            )
            nonzero = np.abs(augmented) > self.zero_frame_threshold
            augmented = np.where(nonzero, augmented + noise, augmented)

        if self.hand_drop_prob > 0:
            for hand_start, hand_end in ((0, 63), (63, 126)):
                drop_mask = np.random.random(augmented.shape[0]) < self.hand_drop_prob
                augmented[drop_mask, hand_start:hand_end] = 0.0

        if self.frame_drop_prob > 0:
            drop_mask = np.random.random(augmented.shape[0]) < self.frame_drop_prob
            augmented[drop_mask] = 0.0

        return augmented

    def __getitem__(self, idx):
        file_path = self.filepaths[idx]
        label = self.labels[idx]

        # Load raw data
        raw_matrix = np.load(file_path).astype(np.float32)

        if self.trim_zero_frames:
            raw_matrix = self._trim_zero_frames(raw_matrix)
        if self.normalize_hands:
            raw_matrix = self._normalize_hands(raw_matrix)

        raw_matrix, usable_frames = self._select_frames(raw_matrix)
        if self.augment:
            raw_matrix = self._augment_sequence(raw_matrix)
        raw_tensor = torch.FloatTensor(raw_matrix)

        # Pad
        padded_tensor = torch.zeros(
            self.max_frames, raw_tensor.shape[1]
        )  # Should be 126 for this dataset

        # Copy to padded tensor
        padded_tensor[:usable_frames, :] = raw_tensor[:usable_frames, :]

        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.long)

        return padded_tensor, label_tensor, usable_frames
