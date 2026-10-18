# coding: utf-8
"""
Loss functions for FLAME model fitting.

This module provides loss functions used during FLAME optimization,
including landmark loss, masked vertex loss, and regularization terms
for shape, expression, and pose parameters.
"""

import torch


def landmark_loss(pred_landmarks: torch.Tensor, target_landmarks: torch.Tensor) -> torch.Tensor:
    """
    Computes Mean Squared Error between predicted and target 3D/2D landmarks.
    Handles slicing if landmark dimensions differ (e.g., 105 vs 68).
    """
    if pred_landmarks.shape[1] == 105 and target_landmarks.shape[1] == 68:
        pred_landmarks = pred_landmarks[:, :68, :]
    return torch.mean((pred_landmarks - target_landmarks) ** 2)


def chamfer_vertex_loss(masked_pred_vertices: torch.Tensor, target_vertices: torch.Tensor) -> torch.Tensor:
    """
    Computes bidirectional Chamfer Distance between masked FLAME face vertices and target vertices.
    """
    dist_matrix = torch.cdist(masked_pred_vertices, target_vertices, p=2)
    min_dist_pred_to_target, _ = torch.min(dist_matrix, dim=2)
    min_dist_target_to_pred, _ = torch.min(dist_matrix, dim=1)


    return torch.mean(min_dist_pred_to_target ** 2) + torch.mean(min_dist_target_to_pred ** 2)


def shape_regularization(betas: torch.Tensor, lambda_reg: float = 1e-4) -> torch.Tensor:
    """
    L2 regularization on shape parameters to encourage plausible shapes.

    Args:
        betas: Shape coefficients, shape (B, n_shape).
        lambda_reg: Regularization weight.

    Returns:
        Regularization loss scalar.
    """
    return lambda_reg * torch.sum(betas ** 2)


def expression_regularization(expression: torch.Tensor, lambda_reg: float = 1e-4) -> torch.Tensor:
    """
    L2 regularization on expression parameters.

    Args:
        expression: Expression coefficients, shape (B, n_exp).
        lambda_reg: Regularization weight.

    Returns:
        Regularization loss scalar.
    """
    return lambda_reg * torch.sum(expression ** 2)


def pose_regularization(pose: torch.Tensor, lambda_reg: float = 1e-3) -> torch.Tensor:
    """
    L2 regularization on pose parameters to avoid extreme rotations.

    Args:
        pose: Pose parameters, shape (B, 6).
        lambda_reg: Regularization weight.

    Returns:
        Regularization loss scalar.
    """
    return lambda_reg * torch.sum(pose ** 2)


def jaw_pose_regularization(pose: torch.Tensor, lambda_reg: float = 1e-2) -> torch.Tensor:
    """
    Strong L2 regularization on jaw pose parameters (indices 3:6) to prevent
    unnatural jaw/chin drift and mouth opening during optimization.

    The jaw pose controls the jaw joint rotation. Without strong regularization,
    the optimizer may open the jaw to reduce landmark or vertex loss, causing
    morphological distortion in the chin area.

    Args:
        pose: Pose parameters, shape (B, 6). Indices 0:3 are head rotation,
              indices 3:6 are jaw rotation.
        lambda_reg: Regularization weight for jaw pose.

    Returns:
        Regularization loss scalar.
    """
    jaw_pose = pose[:, 3:6]  # Jaw rotation parameters
    return lambda_reg * torch.sum(jaw_pose ** 2)
