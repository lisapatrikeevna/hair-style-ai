# coding: utf-8
"""
FLAME (Faces Learned with an Articulated Model and Expressions) parametric model.

This module implements the FLAME 2023 Open model as a PyTorch nn.Module,
including shape blendshapes, expression blendshapes, pose corrective
blendshapes, kinematic tree rigid transforms, and linear blend skinning (LBS).

The FLAME model weight file is loaded from a hardcoded default path
relative to this module's location:
    weights/flame/FLAME2023Open/flame2023_Open.pkl

Reference:
    Li, T., Bolkart, T., Black, M. J., Li, H., & Romero, J. (2017).
    Learning a model of facial shape and expression from 4D scans.
    ACM Transactions on Graphics (Proc. SIGGRAPH Asia), 36(6), 194.
"""

import os
import torch
import torch.nn as nn

from .flame_utils import load_pkl, to_tensor, batch_rodrigues

# Default path to the FLAME model weights, resolved relative to this file.
# From generators/, go up one level to ai_engine/, then into weights/flame/.
_DEFAULT_FLAME_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "weights", "flame", "FLAME2023Open", "flame2023_Open.pkl"
)

class FLAMEModel(nn.Module):
    def __init__(self, n_shape: int = 100, n_exp: int = 50):
        super().__init__()
        self.n_shape = n_shape
        self.n_exp = n_exp

        model_path = _DEFAULT_FLAME_MODEL_PATH
        try:
            data = load_pkl(model_path)
        except Exception as e:
            print(f"[FLAME Model] ERROR loading FLAME model: {e}")
            raise

        # Mean template: (1, V, 3)
        mean_template = to_tensor(data["v_template"].reshape(1, -1, 3))
        self.register_buffer("mean_template", mean_template)
        v_count = mean_template.shape[1]  # 5023

        # --- Shape blendshapes ---
        # shapedirs: (5023, 3, 400) — первые 300 это shape, 300-399 expression
        full_shapedirs = to_tensor(data["shapedirs"])  # (5023, 3, 400)
        n_shape_available = full_shapedirs.shape[2]
        print(f"[FLAME Model] shapedirs total components: {n_shape_available}")

        # Shape: берём первые n_shape компонент (обычно 0-99 из 300)
        shape_part = full_shapedirs[:, :, :n_shape].reshape(v_count * 3, n_shape)
        self.register_buffer("shapedirs_mat", shape_part)  # (V*3, n_shape)

        # Expression: в FLAME 2023 Open лежат в индексах 300-399
        expr_start = 300
        expr_end = min(expr_start + n_exp, n_shape_available)
        actual_n_exp = expr_end - expr_start
        if actual_n_exp < n_exp:
            print(f"[FLAME Model] WARNING: requested n_exp={n_exp} but only {actual_n_exp} available. Padding with zeros.")
        exp_part = full_shapedirs[:, :, expr_start:expr_end].reshape(v_count * 3, actual_n_exp)
        # Если нужно, паддим до n_exp нулями
        if actual_n_exp < n_exp:
            pad = torch.zeros(v_count * 3, n_exp - actual_n_exp)
            exp_part = torch.cat([exp_part, pad], dim=1)
        self.register_buffer("expdirs_mat", exp_part)  # (V*3, n_exp)

        # --- Pose corrective blendshapes ---
        # posedirs: (5023, 3, 36) — оставляем в оригинальной форме для einsum
        posedirs = to_tensor(data["posedirs"])  # (5023, 3, 36)
        self.register_buffer("posedirs", posedirs)  # (V, 3, 36)

        # --- Skinning weights ---
        # weights: (5023, 5)
        self.register_buffer("weights", to_tensor(data["weights"]))

        # --- Joint template positions ---
        joint_template = to_tensor(data["J_regressor"] @ data["v_template"].reshape(-1, 3))
        self.register_buffer("joint_template", joint_template)  # (5, 3)

        # --- Faces ---
        self.register_buffer("faces", to_tensor(data["f"], dtype=torch.long))  # (9976, 3)

        # --- Kinematic tree ---
        kintree = to_tensor(data["kintree_table"], dtype=torch.long)  # (2, 5)
        self.register_buffer("kintree_table", kintree)

    def forward(self, betas: torch.Tensor = None, expression: torch.Tensor = None, pose: torch.Tensor = None) -> torch.Tensor:
        device = self.mean_template.device

        batch_size = 1
        if betas is not None:
            batch_size = betas.shape[0]
        elif expression is not None:
            batch_size = expression.shape[0]
        elif pose is not None:
            batch_size = pose.shape[0]

        if betas is None:
            betas = torch.zeros(batch_size, self.n_shape, device=device)
        if expression is None:
            expression = torch.zeros(batch_size, self.n_exp, device=device)
        if pose is None:
            pose = torch.zeros(batch_size, 6, device=device)

        # 1. Shape + Expression blendshapes
        # shapedirs_mat: (V*3, n_shape), betas: (B, n_shape) -> (B, V*3) -> (B, V, 3)
        shape_blends = torch.matmul(betas, self.shapedirs_mat.T).view(batch_size, -1, 3)
        # expdirs_mat: (V*3, n_exp), expression: (B, n_exp) -> (B, V*3) -> (B, V, 3)
        exp_blends = torch.matmul(expression, self.expdirs_mat.T).view(batch_size, -1, 3)

        v_shaped = self.mean_template + shape_blends + exp_blends  # (B, V, 3)

        # 2. Pose corrective blendshapes (lrotmin)
        num_joints = self.kintree_table.shape[1]  # 5

        # Axis-angle -> rotation matrices для всех джоинтов
        # pose: (B, 6) — первые 3 для root, следующие 3 для jaw
        rot_vecs = torch.zeros(batch_size, num_joints * 3, device=device)
        # Индекс 1 — Голова (Head), отдаем ей первые 3 параметра pose
        rot_vecs[:, 3:6] = pose[:, :3]
        # Индекс 2 — Челюсть (Jaw), отдаем ей следующие 3 параметра pose
        rot_vecs[:, 6:9] = pose[:, 3:6]

        rot_mat = batch_rodrigues(rot_vecs.view(batch_size * num_joints, 3))  # (B*J, 3, 3)
        rot_mat = rot_mat.view(batch_size, num_joints, 3, 3)                  # (B, J, 3, 3)

        # lrotmin: (rot_mat - I) для всех джоинтов кроме root
        # Стандарт FLAME: исключаем joint 0 (root), берём остальные 4
        identity = torch.eye(3, device=device, dtype=rot_mat.dtype)
        rot_diff = (rot_mat[:, 1:] - identity).view(batch_size, -1)  # (B, 4*9) = (B, 36)

        # posedirs: (V, 3, 36) — используем einsum напрямую
        pose_blends = torch.einsum("vjd,bd->bvj", self.posedirs, rot_diff)  # (B, V, 3)

        v_posed = v_shaped + pose_blends  # (B, V, 3)

        # 3. Joint positions
        joints = self.joint_template.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 5, 3)

        # 4. Kinematic tree -> posed transform matrices
        transform_mat = self._batch_rigid_transform(rot_mat, joints, self.kintree_table)  # (B, J, 4, 4)

        # 4b. Rest-pose transform matrices (identity rotations, same joint positions)
        rot_mat_rest = torch.eye(3, device=device, dtype=rot_mat.dtype) \
            .unsqueeze(0).expand(batch_size, num_joints, -1, -1)  # (B, J, 3, 3)
        transform_mat_rest = self._batch_rigid_transform(rot_mat_rest, joints, self.kintree_table)  # (B, J, 4, 4)

        # 5. Linear Blend Skinning (with rest-pose correction)
        W = self.weights.unsqueeze(0).expand(batch_size, -1, -1)  # (B, V, 5)
        v_posed = self._lbs(transform_mat, transform_mat_rest, W, v_posed)  # (B, V, 3)

        return v_posed

    def _batch_rigid_transform(self, rot_mat: torch.Tensor, joints: torch.Tensor, kintree_table: torch.Tensor) -> torch.Tensor:
        batch_size = rot_mat.shape[0]
        num_joints = joints.shape[1]
        device = rot_mat.device

        parents = kintree_table[0].to(dtype=torch.int64)  # (J,)

        # Relative joint positions (from parent to child)
        rel_joints = joints.clone()
        for i in range(num_joints):
            p = parents[i].item()
            if 0 <= p < num_joints:
                rel_joints[:, i] = joints[:, i] - joints[:, p]
            # Root (p < 0 or p >= num_joints): keep absolute position

        # Build local 4x4 transform matrices
        local_mat = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0) \
                        .expand(batch_size, num_joints, -1, -1).clone()  # (B, J, 4, 4)
        local_mat[:, :, :3, :3] = rot_mat
        local_mat[:, :, :3,  3] = rel_joints

        # Compute global transforms via kinematic tree
        global_transforms = []
        for i in range(num_joints):
            p = parents[i].item()
            if 0 <= p < num_joints:
                global_transforms.append(torch.matmul(global_transforms[p], local_mat[:, i]))
            else:
                global_transforms.append(local_mat[:, i])  # Root

        return torch.stack(global_transforms, dim=1)  # (B, J, 4, 4)

    def _lbs(self, transform_mat: torch.Tensor, transform_mat_rest: torch.Tensor,
              weights: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Linear Blend Skinning with rest-pose correction.

        Uses the standard SMPL/FLAME LBS formula:
            v_posed = T_posed_blended @ T_rest_blended^-1 @ v_rest

        This ensures that when pose parameters are zero, the output vertices
        exactly match the input rest-pose positions.

        Args:
            transform_mat: Posed joint transforms (B, J, 4, 4).
            transform_mat_rest: Rest-pose joint transforms (B, J, 4, 4).
            weights: Skinning weights (B, V, J).
            positions: Rest-pose vertex positions (B, V, 3).

        Returns:
            Skinned vertex positions (B, V, 3).
        """
        batch_size, num_verts = positions.shape[:2]
        device = positions.device

        # Blend posed and rest transforms by skinning weights
        weighted_T_posed = torch.einsum("bjki,bvj->bvki", transform_mat, weights)  # (B, V, 4, 4)
        weighted_T_rest = torch.einsum("bjki,bvj->bvki", transform_mat_rest, weights)  # (B, V, 4, 4)

        # Inverse of blended rest transform
        T_rest_inv = torch.inverse(weighted_T_rest)  # (B, V, 4, 4)

        # Combined transform: T_posed @ T_rest^-1
        combined_T = torch.matmul(weighted_T_posed, T_rest_inv)  # (B, V, 4, 4)

        # Apply to vertices in homogeneous coordinates
        ones = torch.ones(batch_size, num_verts, 1, device=device)
        positions_h = torch.cat([positions, ones], dim=2)  # (B, V, 4)
        v_skinned = torch.matmul(combined_T, positions_h.unsqueeze(-1)).squeeze(-1)  # (B, V, 4)

        return v_skinned[:, :, :3]  # (B, V, 3)


