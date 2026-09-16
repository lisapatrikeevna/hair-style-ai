import os
import numpy as np
import mediapipe as mp
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# class FaceParserService:
#     @staticmethod
#     def create_hair_mask(image_path: str) -> Image.Image:
#         # Resolve absolute path to weights relative to this file
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         model_path = os.path.join(current_dir, "weights", "hair_segmenter.tflite")
#
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"Model file not found at: {model_path}")
#
#         # 1. Load image via PIL
#         pil_image = Image.open(image_path).convert("RGB")
#
#         # 2. Convert PIL image to MediaPipe Image format
#         image_np = np.array(pil_image)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
#
#         # 3. Configure segmenter with absolute model path
#         base_options = python.BaseOptions(model_asset_path=model_path)
#         options = vision.ImageSegmenterOptions(
#             base_options=base_options,
#             output_category_mask=True
#         )
#
#         # 4. Run segmentation model
#         with vision.ImageSegmenter.create_from_options(options) as segmenter:
#             segmentation_result = segmenter.segment(mp_image)
#             category_mask = segmentation_result.category_mask.numpy_view()
#
#         # Remove extra 3D dimension from MediaPipe output (H, W, 1) -> (H, W)
#         category_mask_2d = np.squeeze(category_mask)
#
#         # Class ID 1 represents hair in MediaPipe hair_segmenter
#         hair_mask_np = np.where(category_mask_2d == 1, 255, 0).astype(np.uint8)
#
#         return Image.fromarray(hair_mask_np, mode="L")
import os
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceParserService:
    @staticmethod
    def create_hair_mask(image_path: str, dilate_kernel_size: int = 15) -> Image.Image:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "weights", "hair_segmenter.tflite")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        pil_image = Image.open(image_path).convert("RGB")
        image_np = np.array(pil_image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True
        )

        with vision.ImageSegmenter.create_from_options(options) as segmenter:
            segmentation_result = segmenter.segment(mp_image)
            category_mask = segmentation_result.category_mask.numpy_view()

        category_mask_2d = np.squeeze(category_mask)
        hair_mask_np = np.where(category_mask_2d == 1, 255, 0).astype(np.uint8)

        # Dilate mask to cover edges around hair
        if dilate_kernel_size > 0:
            kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
            hair_mask_np = cv2.dilate(hair_mask_np, kernel, iterations=1)

        return Image.fromarray(hair_mask_np, mode="L")