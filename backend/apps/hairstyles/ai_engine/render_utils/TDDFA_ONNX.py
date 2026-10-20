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

from render_utils.io import _load
from render_utils.functions import (
    crop_img, parse_roi_box_from_bbox, parse_roi_box_from_landmark,
)
from render_utils.tddfa_util import _parse_param, similar_transform
from render_utils.bfm.bfm import BFMModel
make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))



class TDDFA_ONNX(object):
    """TDDFA_ONNX: the ONNX version of Three-D Dense Face Alignment (TDDFA)"""

    def __init__(self, **kvs):
        # Helper to resolve path: argument -> container path -> local config default
        def get_path(key, container_path, local_default):
            val = kvs.get(key)
            return val if (val is not None and osp.exists(str(val))) else (
                container_path if osp.exists(container_path) else local_default)

        # config
        self.size = kvs.get('size', 120)
        self.gpu_mode = kvs.get('gpu_mode', False)
        self.gpu_id = kvs.get('gpu_id', 0)


        bfm_fp = get_path('bfm_fp', "/app/weights/bfm_noneck_v3.pkl", make_abs_path('configs/bfm_noneck_v3.pkl'))
        param_mean_std_fp = get_path('param_mean_std_fp', "/app/weights/param_mean_std_62d_120x120.pkl",
                                     make_abs_path('configs/param_mean_std_62d_120x120.pkl'))
        onnx_fp = get_path('onnx_fp', "/app/weights/mb1_120x120.onnx",
                           make_abs_path(f'configs/mb1_{self.size}x{self.size}.onnx'))

        # Direct loading of BFM base data and triangles
        # bfm_data = _load(bfm_fp)
        # self.u_base = bfm_data.get('u')
        # self.w_shp_base = bfm_data.get('w_shp')
        # self.w_exp_base = bfm_data.get('w_exp')

        # tri_npy_path = osp.join(osp.dirname(osp.realpath(__file__)), "configs", "tri.npy")
        # self.tri = np.load(tri_npy_path) if osp.exists(tri_npy_path) else bfm_data.get('tri')[:, :60532]

        bfm = BFMModel(bfm_fp, shape_dim=kvs.get('shape_dim', 40), exp_dim=kvs.get('exp_dim', 10))
        self.tri = bfm.tri

        # Sparse keypoints-only bases (38,365 vertices) for backward compatibility
        self.u_base = bfm.u_base
        self.w_shp_base = bfm.w_shp_base
        self.w_exp_base = bfm.w_exp_base

        # Full dense BFM bases (53,489 vertices) for dense mesh reconstruction
        self.u_dense = bfm.u
        self.w_shp_dense = bfm.w_shp
        self.w_exp_dense = bfm.w_exp

        # Initialize ONNX runtime session
        self.session = onnxruntime.InferenceSession(onnx_fp, None)
        self.bfm_session = None

        # Load parameter normalization configs
        r = _load(param_mean_std_fp)
        self.param_mean = r.get('mean')
        self.param_std = r.get('std')

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

    def recon_vers(self, param_lst, roi_box_lst, **kvs):
        """Reconstruct 3D vertices from predicted parameters.

        Args:
            param_lst: List of predicted parameters.
            roi_box_lst: List of ROI boxes for each face.
            **kvs: Additional keyword arguments.
                - dense_flag (bool): If True, use full dense BFM (53,489 vertices).
                                     If False, use sparse keypoints (38,365 vertices).

        Returns:
            ver_lst: List of 3D vertex arrays with shape (3, num_vertices).
        """
        size = self.size
        dense_flag = kvs.get('dense_flag', False)
        ver_lst = []

        for idx, (param, roi_box) in enumerate(zip(param_lst, roi_box_lst)):
            R, offset, alpha_shp, alpha_exp = _parse_param(param)

            if dense_flag:
                # Dense reconstruction: use full BFM bases (53,489 vertices)
                pts3d = R @ (self.u_dense + self.w_shp_dense @ alpha_shp + self.w_exp_dense @ alpha_exp). \
                    reshape(3, -1, order='F') + offset
            else:
                # Sparse reconstruction: use keypoints-only bases (38,365 vertices)
                pts3d = R @ (self.u_base + self.w_shp_base @ alpha_shp + self.w_exp_base @ alpha_exp). \
                    reshape(3, -1, order='F') + offset

            pts3d = similar_transform(pts3d, roi_box, size)
            ver_lst.append(pts3d)

        return ver_lst
