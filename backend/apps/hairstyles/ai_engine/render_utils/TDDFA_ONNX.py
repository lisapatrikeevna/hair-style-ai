# coding: utf-8

__author__ = 'cleardusk'

import os.path as osp
import os
import glob
import numpy as np
import cv2
import onnxruntime
import sys
sys.stdout.reconfigure(line_buffering=True)

# from render_utils.onnx import convert_to_onnx
from render_utils.io import _load
from render_utils.functions import (
    crop_img, parse_roi_box_from_bbox, parse_roi_box_from_landmark,
)
from render_utils.tddfa_util import _parse_param, similar_transform
# from bfm.bfm import BFMModel
# from bfm.bfm_onnx import convert_bfm_to_onnx

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_noneck_path():
    if os.path.exists("/app/weights/bfm_noneck_v3.pkl"):
        return "/app/weights/bfm_noneck_v3.pkl"

    if os.path.exists("/app/weights/skull_weights/bfm_noneck_v3.pkl"):
        return "/app/weights/skull_weights/bfm_noneck_v3.pkl"

    path_local = os.path.abspath(os.path.join(BASE_DIR, "weights/skull_weights/bfm_noneck_v3.pkl"))
    if not os.path.exists(path_local):
        path_local = os.path.abspath(os.path.join(BASE_DIR, "../weights/skull_weights/bfm_noneck_v3.pkl"))
    return path_local


def get_param_mean_std_path():
    if os.path.exists("/app/weights/param_mean_std_62d_120x120.pkl"):
        return "/app/weights/param_mean_std_62d_120x120.pkl"

    if os.path.exists("/app/weights/skull_weights/param_mean_std_62d_120x120.pkl"):
        return "/app/weights/skull_weights/param_mean_std_62d_120x120.pkl"

    path_local = os.path.abspath(os.path.join(BASE_DIR, "weights/skull_weights/param_mean_std_62d_120x120.pkl"))
    if not os.path.exists(path_local):
        path_local = os.path.abspath(os.path.join(BASE_DIR, "../weights/skull_weights/param_mean_std_62d_120x120.pkl"))
    return path_local


def get_main_onnx_path():
    if os.path.exists("/app/weights/mb1_120x120.onnx"):
        return "/app/weights/mb1_120x120.onnx"

    if os.path.exists("/app/weights/skull_weights/mb1_120x120.onnx"):
        return "/app/weights/skull_weights/mb1_120x120.onnx"

    path_local = os.path.abspath(os.path.join(BASE_DIR, "weights/skull_weights/mb1_120x120.onnx"))
    if not os.path.exists(path_local):
        path_local = os.path.abspath(os.path.join(BASE_DIR, "../weights/skull_weights/mb1_120x120.onnx"))
    return path_local


class TDDFA_ONNX(object):
    """TDDFA_ONNX: the ONNX version of Three-D Dense Face Alignment (TDDFA)"""

    def __init__(self, **kvs):
        # 1. Проверяем все возможные пути внутри контейнера (с учетом затирания папок монтированием)
        possible_bfm_paths = [
            "/app/weights/bfm_noneck_v3.pkl",
            "/app/weights/skull_weights/bfm_noneck_v3.pkl",
            "/app/apps/hairstyles/ai_engine/weights/skull_weights/bfm_noneck_v3.pkl",
            osp.join(osp.dirname(__file__), "bfm_noneck_v3.pkl")
        ]

        possible_param_paths = [
            "/app/weights/param_mean_std_62d_120x120.pkl",
            "/app/weights/skull_weights/param_mean_std_62d_120x120.pkl",
            "/app/apps/hairstyles/ai_engine/weights/skull_weights/param_mean_std_62d_120x120.pkl",
            osp.join(osp.dirname(__file__), "param_mean_std_62d_120x120.pkl")
        ]

        possible_onnx_paths = [
            "/app/weights/mb1_120x120.onnx",
            "/app/weights/skull_weights/mb1_120x120.onnx",
            "/app/apps/hairstyles/ai_engine/weights/skull_weights/mb1_120x120.onnx",
            osp.join(osp.dirname(__file__), "mb1_120x120.onnx")
        ]

        # Выбираем первый существующий файл из списков
        bfm_fp = next((p for p in possible_bfm_paths if os.path.exists(p)), possible_bfm_paths[0])
        param_mean_std_fp = next((p for p in possible_param_paths if os.path.exists(p)), possible_param_paths[0])
        onnx_fp = next((p for p in possible_onnx_paths if os.path.exists(p)), possible_onnx_paths[0])

        print(f"[DEBUG INITIALIZATION] Selected bfm_fp: {bfm_fp} (Exists: {os.path.exists(bfm_fp)})")
        print(
            f"[DEBUG INITIALIZATION] Selected param_fp: {param_mean_std_fp} (Exists: {os.path.exists(param_mean_std_fp)})")
        print(f"[DEBUG INITIALIZATION] Selected onnx_fp: {onnx_fp} (Exists: {os.path.exists(onnx_fp)})")

        # 2. Загружаем базовую 3D-модель лица
        if os.path.exists(bfm_fp):
            bfm_data = _load(bfm_fp)
            self.u_base = bfm_data.get('u')
            self.w_shp_base = bfm_data.get('w_shp')
            self.w_exp_base = bfm_data.get('w_exp')
            self.tri = bfm_data.get('tri')
        else:
            raise FileNotFoundError(f"[Error] Сritical file not found at: {bfm_fp}")

        self.gpu_mode = kvs.get('gpu_mode', False)
        self.gpu_id = kvs.get('gpu_id', 0)
        self.size = kvs.get('size', 120)

        # 3. Загружаем параметры нормализации
        if os.path.exists(param_mean_std_fp):
            r = _load(param_mean_std_fp)
            self.param_mean = r.get('mean')
            self.param_std = r.get('std')
        else:
            raise FileNotFoundError(f"[Error] Normalization file not found at: {param_mean_std_fp}")

        # 4. Инициализируем основную сессию ONNX
        if not os.path.exists(onnx_fp):
            raise FileNotFoundError(f"[Error] ONNX model not found at: {onnx_fp}")

        self.session = onnxruntime.InferenceSession(onnx_fp, None)
        self.bfm_session = None

    def __call__(self, img_ori, objs, **kvs):
        # Crop image, forward to get the param
        param_lst = []
        roi_box_lst = []

        crop_policy = kvs.get('crop_policy', 'box')
        for obj in objs:
            if crop_policy == 'box':
                # by face box
                roi_box = parse_roi_box_from_bbox(obj)
            elif crop_policy == 'landmark':
                # by landmarks
                roi_box = parse_roi_box_from_landmark(obj)
            else:
                raise ValueError(f'Unknown crop policy {crop_policy}')

            roi_box_lst.append(roi_box)
            img = crop_img(img_ori, roi_box)
            img = cv2.resize(img, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
            img = (img - 127.5) / 128.

            inp_dct = {'input': img}

            param = self.session.run(None, inp_dct)[0]
            param = param.flatten().astype(np.float32)
            param = param * self.param_std + self.param_mean  # re-scale
            param_lst.append(param)

        return param_lst, roi_box_lst

    def get_depth_map(ver, h, w):
        """
        Превращает 3D вертексы в объемную карту глубины (светотень)
        """
        z = ver[2, :]

        z_min, z_max = np.min(z), np.max(z)
        if z_max - z_min > 0:
            z_norm = (z - z_min) / (z_max - z_min) * 255
        else:
            z_norm = z

        z_norm = z_norm.astype(np.uint8)
        depth_img = np.zeros((h, w), dtype=np.uint8)
        pts2d = ver[:2, :].T.astype(np.int32)
        idx = np.argsort(z)

        for i in idx:
            x_coord = pts2d[i, 0]
            y_coord = pts2d[i, 1]
            if 0 <= x_coord < w and 0 <= y_coord < h:
                depth_img[y_coord, x_coord] = z_norm[i]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        depth_img = cv2.morphologyEx(depth_img, cv2.MORPH_CLOSE, kernel)
        depth_img = cv2.GaussianBlur(depth_img, (7, 7), 0)

        return depth_img

    def recon_vers(self, param_lst, roi_box_lst, **kvs):
        size = self.size
        ver_lst = []

        for param, roi_box in zip(param_lst, roi_box_lst):
            R, offset, alpha_shp, alpha_exp = _parse_param(param)

            pts3d = R @ (self.u_base + self.w_shp_base @ alpha_shp + self.w_exp_base @ alpha_exp). \
                reshape(3, -1, order='F') + offset

            pts3d = similar_transform(pts3d, roi_box, size)
            ver_lst.append(pts3d)

        return ver_lst
