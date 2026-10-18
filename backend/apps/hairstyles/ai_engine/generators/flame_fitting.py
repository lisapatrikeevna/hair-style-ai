# coding: utf-8
"""
FLAME (Faces Learned with an Articulated Model and Expressions) fitting pipeline.

This module orchestrates the PyTorch-based optimization loop that fits the FLAME
parametric head model to 3D target vertices obtained from 3DDFA_V2.
It optimizes shape (beta), expression (exp), and pose (rotation) parameters
using landmark loss and multi-stage dense vertex loss.

All weight file paths are managed internally by each submodule
(flame_model, flame_landmarks), so this orchestrator does not handle paths.

The module delegates to submodules for model definition, loss computation,
landmark/mask loading, and mesh export.

Reference:
    Li, T., Bolkart, T., Black, M. J., Li, H., & Romero, J. (2017).
    Learning a model of facial shape and expression from 4D scans.
    ACM Transactions on Graphics (Proc. SIGGRAPH Asia), 36(6), 194.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import traceback

from .flame_model import FLAMEModel
from .flame_landmarks import LandmarkEmbedder, FLAMEMaskLoader
from .flame_loss import (
    landmark_loss,
    shape_regularization,
    expression_regularization,
    pose_regularization,
    jaw_pose_regularization,
    chamfer_vertex_loss,
)
from .flame_exporter import save_obj_mesh
from .flame_texture import project_texture_fast, generate_uv_coordinates
from .flame_utils import (
    save_step1_debug_visualization,
    save_debug_combined_obj,
    _save_debug_image,
    _project_3d_to_2d,
    _load_image,
)


# ---------------------------------------------------------------------------
# Main fitting function
# ---------------------------------------------------------------------------

def fit_flame_to_vertices(target_vertices: np.ndarray, ver_landmarks: np.ndarray, rotation_matrix: np.ndarray,
                          output_obj_path: str, image_path: str = None) -> dict:
    """
    Fit FLAME model to target vertices using PyTorch multi-stage optimization.

    Args:
        target_vertices: Target vertices from 3DDFA_V2, shape (3, N).
        ver_landmarks: Target landmarks from 3DDFA_V2, shape (3, 68).
        rotation_matrix: Rotation/affine matrix from 3DDFA_V2, shape (3, 3).
        output_obj_path: Path to save the fitted mesh as .obj file.
        image_path: Optional path to the original face image (reserved for final texturing step).

    Returns:
        Dictionary containing fitted vertices, faces, and optimized parameters.
    """
    n_shape: int = 100
    n_exp: int = 50
    num_iterations: int = 1000

    # Loss weights — scheduled per stage to prevent morphological distortion
    # Stage 1 (0-199): Pose only, frozen shape/expression
    # Stage 2 (200-599): Unfreeze shape/expression with strong regularization
    # Stage 3 (600-999): Dense mesh fine-tuning with refined mask
    lambda_landmark_stage1: float = 1.0
    lambda_landmark_stage3: float = 0.3  # Reduced in Stage 3 to avoid overfitting landmarks
    lambda_vertex: float = 0.5  # Enabled on Stage 3 for dense mesh fine-tuning
    lambda_shape_reg_stage1: float = 1e-4  # Low during pose-only stage
    lambda_shape_reg_stage2: float = 3e-3  # Strong regularization to prevent shape distortion
    lambda_exp_reg_stage1: float = 1e-4
    lambda_exp_reg_stage2: float = 5e-3  # Strong regularization to prevent expression distortion
    lambda_pose_reg: float = 1e-3
    lambda_jaw_reg: float = 3e-2  # Strong jaw regularization to prevent chin drift and mouth opening

    # Determine device — prefer CPU for stability during optimization.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FLAME Fitting] Using device: {device}")

    debug_dir = os.path.join(os.path.dirname(output_obj_path), "debug_flame") if output_obj_path else "debug_flame"
    os.makedirs(debug_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load the FLAME model
    # ------------------------------------------------------------------
    try:
        flame = FLAMEModel(n_shape=n_shape, n_exp=n_exp).to(device)
    except Exception as e:
        print(f"[FLAME Fitting] ERROR loading FLAME model: {e}")
        traceback.print_exc()
        raise

    # ------------------------------------------------------------------
    # 2. Load landmark embedder
    # ------------------------------------------------------------------
    landmarker = None
    try:
        landmarker = LandmarkEmbedder()
    except Exception as e:
        print(f"[FLAME Fitting] ERROR loading landmarks: {e}. Landmark loss will be skipped.")

    # ------------------------------------------------------------------
    # 3. Load vertex masks (face mask refined to exclude lips for Chamfer loss)
    # ------------------------------------------------------------------
    try:
        mask_loader = FLAMEMaskLoader()
        # FLAMEMaskLoader returns boolean tensors of shape (5023,)
        face_mask_bool = mask_loader.get_mask("face").to(device=device)
        lips_mask_bool = mask_loader.get_mask("lips").to(device=device)

        # Refined face mask: face vertices excluding lips vertices
        # Use boolean subtraction: face AND (NOT lips)
        refined_face_mask_bool = face_mask_bool & (~lips_mask_bool)

        # Convert boolean masks to index tensors for advanced indexing
        face_mask = torch.nonzero(face_mask_bool, as_tuple=False).squeeze(-1)
        refined_face_mask = torch.nonzero(refined_face_mask_bool, as_tuple=False).squeeze(-1)

        print(f"[FLAME Fitting] Face mask: {len(face_mask)} vertices, "
              f"Lips mask: {lips_mask_bool.sum().item()} vertices, "
              f"Refined face mask (excluding lips): {len(refined_face_mask)} vertices")
    except Exception as e:
        print(f"[FLAME Fitting] ERROR loading masks: {e}. Using all-ones fallback.")
        face_mask = torch.arange(5023, dtype=torch.long, device=device)
        refined_face_mask = face_mask  # Fallback: use full face mask

    if target_vertices.shape[0] == 3:
        target_vertices_np = target_vertices.T
    else:
        target_vertices_np = target_vertices.copy()

    if ver_landmarks.shape[0] == 3:
        target_landmarks_np = ver_landmarks.T
    else:
        target_landmarks_np = ver_landmarks.copy()

    # ------------------------------------------------------------------
    # 4. Alignment, Scaling & Step 1 Debug Blocks
    # ------------------------------------------------------------------

    # --- Step 4a: Генерация и сохранение чистой канонической болванки FLAME ---
    with torch.no_grad():
        test_zero_pose = torch.zeros(1, 6, device=device)
        test_zero_betas = torch.zeros(1, n_shape, device=device)
        test_zero_exp = torch.zeros(1, n_exp, device=device)

        raw_flame_tensor = flame(betas=test_zero_betas, expression=test_zero_exp, pose=test_zero_pose)
        raw_flame_v = raw_flame_tensor.squeeze(0).cpu().numpy()
        raw_flame_f = flame.faces.cpu().numpy()

        raw_flame_v = raw_flame_v - raw_flame_v.mean(axis=0)

        pure_flame_path = os.path.join(debug_dir, "pure_canonical_flame.obj")
        save_obj_mesh(pure_flame_path, raw_flame_v, raw_flame_f)
        print(f"[FLAME Test] Pure canonical FLAME saved to: {pure_flame_path}")

    # Получаем канонические ландмарки FLAME для выравнивания
    with torch.no_grad():
        zero_pose = torch.zeros(1, 6, device=device)
        zero_betas = torch.zeros(1, n_shape, device=device)
        zero_exp = torch.zeros(1, n_exp, device=device)
        canonical_mesh = flame(betas=zero_betas, expression=zero_exp, pose=zero_pose)

        if landmarker is not None:
            canonical_landmarks_tensor = landmarker.get_landmarks(canonical_mesh, flame.faces)
            canonical_landmarks_np = canonical_landmarks_tensor.detach().cpu().numpy().squeeze(0)
        else:
            canonical_landmarks_np = canonical_mesh[0, :68, :].detach().cpu().numpy()

    # --- Step 4b: Применение поворота от 3DDFA_V2 ---
    if rotation_matrix is not None and rotation_matrix.shape == (3, 3):
        target_vertices_np = target_vertices_np @ rotation_matrix.T
        target_landmarks_np = target_landmarks_np @ rotation_matrix.T
        print(f"[FLAME Fitting] Applied 3DDFA_V2 rotation matrix to target coordinates.")
    else:
        print("[FLAME Fitting] WARNING: No valid rotation_matrix provided, skipping rotation.")

    # --- Step 4c: Мульти-якорный расчет масштаба и жесткое выравнивание (Kabsch / SVD) ---
    anchor_pairs = [
        (36, 45),  # Внешние уголки глаз
        (39, 42),  # Внутренние уголки глаз
        (31, 35),  # Крылья носа
        (48, 54),  # Уголки рта
        (30, 8)    # Кончик носа — подбородок
    ]

    calculated_scales = []
    for p1, p2 in anchor_pairs:
        target_dist = np.linalg.norm(target_landmarks_np[p1] - target_landmarks_np[p2])
        canonical_dist = np.linalg.norm(canonical_landmarks_np[p1] - canonical_landmarks_np[p2])
        if target_dist > 1e-6:
            calculated_scales.append(canonical_dist / target_dist)

    if calculated_scales:
        metric_scale = np.median(calculated_scales)
        target_landmarks_np = target_landmarks_np * metric_scale
        target_vertices_np = target_vertices_np * metric_scale
        print(f"[FLAME Fitting] Multi-anchor median scale applied: {metric_scale:.6f} "
              f"(based on {len(calculated_scales)} segments)")
    else:
        print("[FLAME Fitting] WARNING: Multi-anchor scaling failed, fallback to 1.0.")

    src_centroid = target_landmarks_np.mean(axis=0)
    tgt_centroid = canonical_landmarks_np.mean(axis=0)

    src_centered = target_landmarks_np - src_centroid
    tgt_centered = canonical_landmarks_np - tgt_centroid

    H = src_centered.T @ tgt_centered
    U, S, Vt = np.linalg.svd(H)
    R_align = Vt.T @ U.T

    if np.linalg.det(R_align) < 0:
        Vt[-1, :] *= -1
        R_align = Vt.T @ U.T

    t_align = canonical_landmarks_np[30] - (target_landmarks_np[30] @ R_align.T)
    print(f"[FLAME Fitting] Rigid alignment computed -> Translation: {t_align}")

    vertices_aligned = (target_vertices_np @ R_align.T) + t_align
    landmarks_aligned = (target_landmarks_np @ R_align.T) + t_align

    target_vertices_tensor = torch.tensor(vertices_aligned, dtype=torch.float32, device=device).unsqueeze(0)
    target_landmarks_tensor = torch.tensor(landmarks_aligned, dtype=torch.float32, device=device).unsqueeze(0)

    # --- Step 4d: Сохранение 2D-визуализации сопоставления точек Шага 1 ---
    try:
        save_step1_debug_visualization(
            aligned_landmarks=landmarks_aligned,
            canonical_landmarks=canonical_landmarks_np,
            output_path=os.path.join(debug_dir, "debug_step1_alignment.png")
        )
    except Exception as e:
        print(f"[FLAME Fitting] WARNING: Failed to save Step 1 debug image: {e}")

    # ------------------------------------------------------------------
    # 5. Initialize optimization parameters
    # ------------------------------------------------------------------
    flame_betas = torch.zeros(1, n_shape, dtype=torch.float32, device=device, requires_grad=True)
    flame_expression = torch.zeros(1, n_exp, dtype=torch.float32, device=device, requires_grad=True)
    flame_pose = torch.zeros(1, 6, dtype=torch.float32, device=device, requires_grad=True)
    flame_trans = torch.zeros(1, 3, dtype=torch.float32, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([flame_pose, flame_trans], lr=1e-2)

    # ------------------------------------------------------------------
    # 6. Multi-stage optimization loop
    # ------------------------------------------------------------------
    # Stage 1 (0-199): Global pose + translation only (shape/expression frozen)
    # Stage 2 (200-599): Unfreeze shape + expression with strong regularization
    # Stage 3 (600-999): Dense mesh fine-tuning with refined face mask (no lips)
    print("[FLAME Fitting] Stage 1: Optimizing global pose and translation...")
    best_loss = float('inf')

    for iter_idx in range(num_iterations):
        try:
            # --- Stage transitions ---
            if iter_idx == 200:
                print("[FLAME Fitting] Stage 2: Unfreezing shape and expression with strong regularization...")
                optimizer = torch.optim.Adam(
                    [flame_pose, flame_trans, flame_betas, flame_expression], lr=2e-3
                )

            if iter_idx == 600:
                print("[FLAME Fitting] Stage 3: Fine-tuning dense mesh with refined face mask (excluding lips)...")
                optimizer = torch.optim.Adam(
                    [flame_pose, flame_trans, flame_betas, flame_expression], lr=5e-4
                )

            optimizer.zero_grad()

            pred_vertices = flame(betas=flame_betas, expression=flame_expression,
                                  pose=flame_pose) + flame_trans.unsqueeze(1)

            if landmarker is not None:
                pred_landmarks = landmarker.get_landmarks(pred_vertices, flame.faces)
            else:
                pred_landmarks = pred_vertices[:, :68, :]

            # --- Compute losses with stage-based scheduling ---
            loss_landmark = landmark_loss(pred_landmarks, target_landmarks_tensor)

            # Regularization weights vary by stage
            if iter_idx < 200:
                # Stage 1: pose only, low regularization
                lambda_landmark = lambda_landmark_stage1
                lambda_shape = lambda_shape_reg_stage1
                lambda_exp = lambda_exp_reg_stage1
            elif iter_idx < 600:
                # Stage 2: strong shape/expression regularization to prevent distortion
                lambda_landmark = lambda_landmark_stage1
                lambda_shape = lambda_shape_reg_stage2
                lambda_exp = lambda_exp_reg_stage2
            else:
                # Stage 3: reduced landmark weight, strong regularization maintained
                lambda_landmark = lambda_landmark_stage3
                lambda_shape = lambda_shape_reg_stage2
                lambda_exp = lambda_exp_reg_stage2

            loss_shape_reg = shape_regularization(flame_betas, lambda_shape)
            loss_exp_reg = expression_regularization(flame_expression, lambda_exp)
            loss_pose_reg = pose_regularization(flame_pose, lambda_pose_reg)
            loss_jaw_reg = jaw_pose_regularization(flame_pose, lambda_jaw_reg)

            # Chamfer loss only in Stage 3, using refined face mask (excluding lips)
            if iter_idx >= 600:
                masked_flame = pred_vertices[:, refined_face_mask, :]
                loss_vertex = chamfer_vertex_loss(masked_flame, target_vertices_tensor)
            else:
                loss_vertex = torch.tensor(0.0, device=device)

            total_loss = (
                loss_landmark * lambda_landmark
                + loss_vertex * lambda_vertex
                + loss_shape_reg
                + loss_exp_reg
                + loss_pose_reg
                + loss_jaw_reg
            )

            total_loss.backward()
            optimizer.step()

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"[FLAME Fitting] NaN detected at iteration {iter_idx + 1}. Breaking loop.")
                break

            current_loss = total_loss.item()
            if current_loss < best_loss:
                best_loss = current_loss

            if iter_idx % 100 == 0 or iter_idx == 0:
                print(
                    f"Iter {iter_idx:04d} | Total Loss: {current_loss:.6f} | "
                    f"Lmk: {loss_landmark.item():.6f} | Mesh: {loss_vertex.detach().item():.6f} | "
                    f"Jaw reg: {loss_jaw_reg.item():.6f}"
                )

        except Exception as e:
            print(f"[FLAME Fitting] ERROR at iteration {iter_idx}: {e}")
            traceback.print_exc()
            break

    # ------------------------------------------------------------------
    # 7. Extract final results & Export OBJ (Volumetric Skull)
    # ------------------------------------------------------------------
    with torch.no_grad():
        final_raw = flame(betas=flame_betas, expression=flame_expression, pose=flame_pose) + flame_trans.unsqueeze(1)
        final_vertices = final_raw.squeeze(0).detach().cpu().numpy()
        final_faces = flame.faces.detach().cpu().numpy()

    result = {
        "vertices": final_vertices,
        "faces": final_faces,
        "betas": flame_betas.detach().cpu().numpy(),
        "expression": flame_expression.detach().cpu().numpy(),
        "pose": flame_pose.detach().cpu().numpy(),
        "trans": flame_trans.detach().cpu().numpy(),
    }

    # ------------------------------------------------------------------
    # 7a. Save the untextured volumetric skull mesh
    # ------------------------------------------------------------------
    save_obj_mesh(output_obj_path, result["vertices"], result["faces"])
    print(f"[FLAME Fitting] Volumetric skull mesh successfully saved to: {output_obj_path}")

    # ------------------------------------------------------------------
    # 7b. Step 5: Image-based UV Texture Projection
    # ------------------------------------------------------------------
    texture_path = None
    uv_coords = None
    textured_obj_path = None

    if image_path is not None and os.path.exists(image_path):
        print("[FLAME Fitting] Step 5: Projecting image texture onto fitted mesh...")

        # Build the combined rotation matrix from the fitted pose parameters
        # The pose tensor contains head rotation (indices 0:3) as axis-angle
        pose_np = result["pose"]  # (1, 6)
        head_axis_angle = pose_np[0, :3]  # Head rotation axis-angle

        # Convert axis-angle to rotation matrix using Rodrigues' formula
        combined_rotation = _axis_angle_to_matrix(head_axis_angle)

        # Use the fitted translation
        fitted_trans = result["trans"].flatten()  # (3,)

        # Generate UV coordinates
        uv_coords, _ = generate_uv_coordinates(result["vertices"], result["faces"])

        # Project the texture
        texture_output_path = os.path.join(
            os.path.dirname(output_obj_path) if output_obj_path else ".",
            "texture.png"
        )

        texture_path = project_texture_fast(
            vertices=result["vertices"],
            faces=result["faces"],
            image_path=image_path,
            rotation_matrix=combined_rotation,
            translation=fitted_trans,
            output_texture_path=texture_output_path,
            texture_size=1024,
        )

        # If texture was successfully created, save a textured OBJ + MTL
        if texture_path is not None and os.path.exists(texture_path):
            textured_obj_path = os.path.splitext(output_obj_path)[0] + "_textured.obj"
            save_obj_mesh(
                textured_obj_path, result["vertices"], result["faces"],
                uv_coords=uv_coords, texture_path=texture_path
            )
            print(f"[FLAME Fitting] Textured mesh saved to: {textured_obj_path}")
            print(f"[FLAME Fitting] Texture map saved to: {texture_path}")
        else:
            print("[FLAME Fitting] WARNING: Texture projection failed, saving untextured mesh only.")
    else:
        print(f"[FLAME Fitting] No image found at {image_path}, skipping texture projection.")

    result["uv_coords"] = uv_coords
    result["texture_path"] = texture_path

    return result



# ---------------------------------------------------------------------------
# Helper: axis-angle to rotation matrix (Rodrigues' formula)
# ---------------------------------------------------------------------------

def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """
    Convert an axis-angle vector to a 3x3 rotation matrix using Rodrigues' formula.

    Args:
        axis_angle: Axis-angle vector, shape (3,).

    Returns:
        3x3 rotation matrix.
    """
    theta = np.linalg.norm(axis_angle)
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)

    axis = axis_angle / theta
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = axis

    # Skew-symmetric cross-product matrix
    K = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ], dtype=np.float32)

    R = np.eye(3, dtype=np.float32) + s * K + (1 - c) * (K @ K)
    return R


# ---------------------------------------------------------------------------
# Convenience wrapper for integration with face_geometry.py
# ---------------------------------------------------------------------------

def fit_flame_pipeline(target_vertices: np.ndarray, rotation: np.ndarray, output_obj_path: str,
                       image_path: str = None, **kwargs) -> bool:
    """
    High-level wrapper for FLAME fitting, designed to integrate with
    the FaceGeometryBuilder.volumetric_skull_geometry() method.
    """
    try:
        result = fit_flame_to_vertices(
            target_vertices=target_vertices,
            ver_landmarks=kwargs.pop("ver_landmarks", None),
            rotation_matrix=rotation,
            output_obj_path=output_obj_path,
            image_path=image_path,
            **kwargs,
        )
        if result and "vertices" in result and "faces" in result:
            vertices = result['vertices']
            faces = result['faces']
            has_nan = np.isnan(vertices).any() or np.isinf(vertices).any()
            print(f"[FLAME Fitting] Final mesh: {vertices.shape[0]} vertices, "
                  f"{faces.shape[0]} faces, NaN={has_nan}")
            return True
        return False
    except Exception as e:
        print(f"[Error] FLAME fitting failed: {e}")
        traceback.print_exc()
        return False
