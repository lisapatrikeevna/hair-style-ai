import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceParserService:
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"

    @classmethod
    def _ensure_model_exists(cls, model_path: str):
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print(f"[Info] Downloading selfie_multiclass model to: {model_path}...")
            urllib.request.urlretrieve(cls.MODEL_URL, model_path)
            print("[Info] Download complete!")

    @classmethod
    def create_hair_mask(cls, image_path: str, dilate_kernel_size: int = 5, debug_dir: str = "media/debug") -> Image.Image:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "weights", "selfie_multiclass_256x256.tflite")

        cls._ensure_model_exists(model_path)

        os.makedirs(debug_dir, exist_ok=True)

        pil_image = Image.open(image_path).convert("RGB")
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image at: {image_path}")

        orig_h, orig_w = image_bgr.shape[:2]

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

        if category_mask_2d.shape != (orig_h, orig_w):
            category_mask_2d = cv2.resize(
                category_mask_2d.astype(np.uint8),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST
            )

        # Классы MediaPipe Selfie Multiclass:
        # 0 - Background (Фон)
        # 1 - Hair (Волосы)
        # 2 - Body-skin (Кожа)
        # 3 - Face-skin (Лицо)
        # 4 - Clothes (Одежда)
        # 5 - Others (Аксессуары)

        # --- ИНВЕРСНЫЙ ПОДХОД ---
        excluded_zones = (
            (category_mask_2d == 0) |  # Фон
            (category_mask_2d == 2) |  # Кожа
            (category_mask_2d == 3) |  # Лицо
            (category_mask_2d == 4)    # Одежда
        )

        hair_mask_np = np.where(excluded_zones, 0, 255).astype(np.uint8)

        if dilate_kernel_size > 0:
            kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
            hair_mask_np = cv2.dilate(hair_mask_np, kernel, iterations=1)

        cv2.imwrite(os.path.join(debug_dir, "1_hair_segmentation.png"), hair_mask_np)

        combined_overlay = cv2.addWeighted(
            image_bgr, 0.6,
            cv2.cvtColor(hair_mask_np, cv2.COLOR_GRAY2BGR), 0.4, 0
        )
        cv2.imwrite(os.path.join(debug_dir, "3_base_inpainting_mask.png"), combined_overlay)

        print(f"[Success] Inverse hair mask saved in '{debug_dir}/'")
        return Image.fromarray(hair_mask_np, mode="L")


if __name__ == "__main__":
    test_photo = "media/test_face.jpg"
    FaceParserService.create_hair_mask(test_photo)


 # python apps/hairstyles/ai_engine/face_parser.py