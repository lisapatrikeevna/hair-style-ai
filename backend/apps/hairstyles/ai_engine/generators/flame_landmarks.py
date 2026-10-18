# coding: utf-8
"""
FLAME landmark embeddings and vertex masks.

This module provides classes for loading FLAME landmark embeddings
(MediaPipe/FLAME mapping) and vertex masks (face, eye, mouth regions)
used during optimization.

Each class resolves its own weight file path relative to this module's
location, so callers do not need to pass file paths.
"""
# Default paths resolved relative to this file's directory.
# From generators/, go up one level to ai_engine/, then into weights/flame/.
import os
import pickle
import numpy as np
import torch

# Default paths resolved relative to this file's directory.
_HERE = os.path.dirname(os.path.abspath(__file__))

_FLAME_2023_68_EMBEDDING_PATH = os.path.join(
    _HERE, "..", "weights", "flame", "FLAME2023Open", "flame_static_embedding_68.pkl"
)

_DEFAULT_MASKS_PATH = os.path.join(
    _HERE, "..", "weights", "flame", "FLAME_masks", "FLAME_masks.pkl"
)


def load_pkl(path: str):
    """
    Utility function to safely load pkl configuration files.
    """
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


class LandmarkEmbedder:
    """
    Extracts 68 facial landmarks from FLAME 2023 Open mesh using barycentric
    interpolation on the static embedding triangles.

    The embedding data is loaded from flame_static_embedding_68.pkl which contains:
      - lmk_face_idx: (68,) triangle indices into the FLAME faces array
      - lmk_b_coords: (68, 3) barycentric weights for each landmark

    Formula:
        triangles = vertices[:, faces[lmk_face_idx]]   # (B, 68, 3, 3)
        landmarks = sum(triangles * b_coords)           # (B, 68, 3)
    """

    def __init__(self, landmarks_path: str = None):
        path = landmarks_path or _FLAME_2023_68_EMBEDDING_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[FLAME Landmarks] CRITICAL ERROR: Embedding file not found at: {path}."
            )

        data = load_pkl(path)

        # Triangle (face) indices for each of the 68 landmarks
        self.lmk_face_idx = np.asarray(data["lmk_face_idx"], dtype=np.int64)  # (68,)

        # Barycentric weights for each landmark on its triangle
        self.lmk_b_coords = np.asarray(data["lmk_b_coords"], dtype=np.float32)  # (68, 3)

        # Store as PyTorch buffers (registered later via get_landmarks)
        self._b_coords_tensor = torch.tensor(self.lmk_b_coords, dtype=torch.float32)  # (68, 3)
        self._face_idx_tensor = torch.tensor(self.lmk_face_idx, dtype=torch.long)     # (68,)

    def get_landmarks(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """
        Extract 68 landmark positions from mesh vertices using barycentric interpolation.

        Args:
            vertices: FLAME mesh vertices, shape (B, V, 3).
            faces: FLAME mesh faces (triangles), shape (F, 3) with vertex indices.

        Returns:
            Landmark positions, shape (B, 68, 3).
        """
        device = vertices.device

        # Get the face indices for each landmark and gather the corresponding triangles
        face_idx = self._face_idx_tensor.to(device=device)
        b_coords = self._b_coords_tensor.to(device=device)  # (68, 3)

        # Gather the three vertices of each triangle: (B, 68, 3, 3)
        triangles = vertices[:, faces[face_idx], :]  # (B, 68, 3, 3)

        # Barycentric interpolation: weighted sum of triangle vertices
        # b_coords: (68, 3) -> unsqueeze for broadcast: (1, 68, 3, 1)
        landmarks = (triangles * b_coords.unsqueeze(-1)).sum(dim=2)  # (B, 68, 3)

        return landmarks


class FLAMEMaskLoader:
    """
    Loads FLAME vertex masks (e.g., face-only, eye, mouth regions)
    for use in masked vertex loss during optimization.
    """

    def __init__(self, masks_path: str = None):
        """
        Args:
            masks_path: Optional path to override the default vertex masks
                file location. If None, uses the built-in default path.
        """
        self.masks = {}
        path = masks_path or _DEFAULT_MASKS_PATH
        if os.path.exists(path):
            try:
                data = load_pkl(path)
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        # The pkl stores index arrays (e.g., [16, 17, 18, ...]).
                        # Create a full 5023-sized boolean mask with True at those indices.
                        full_mask = torch.zeros(5023, dtype=torch.bool)
                        indices = torch.tensor(value.flatten(), dtype=torch.long)
                        full_mask[indices] = True
                        self.masks[key] = full_mask
            except Exception as e:
                print(f"[FLAME Masks] ERROR loading masks: {e}. Using fallback all-ones face mask.")
                self.masks["face"] = torch.ones(5023, dtype=torch.bool)
        else:
            print(f"[FLAME Masks] WARNING: Mask file not found at {path}. Using fallback all-ones face mask.")
            self.masks["face"] = torch.ones(5023, dtype=torch.bool)

    def get_mask(self, mask_name: str) -> torch.Tensor:
        """
        Get a boolean mask for a specific region.

        Args:
            mask_name: Name of the mask (e.g., "face", "eye", "mouth").

        Returns:
            Boolean tensor of shape (V...).
        """
        if mask_name in self.masks:
            return self.masks[mask_name]
        return torch.ones(5023, dtype=torch.bool)

