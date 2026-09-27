# # import os
# # import sys
# # import cv2
# # import numpy as np
# # import mediapipe as mp
# #
# #
# # def generate_skull_base(input_path: str, output_path: str, weights_dir: str):
# #     image = cv2.imread(input_path)
# #     if image is None:
# #         print(f"[Error] Failed to read input image: {input_path}")
# #         sys.exit(1)
# #
# #     h, w, _ = image.shape
# #
# #     # Инициализируем MediaPipe Face Mesh для поиска 3D-точек лица
# #     mp_face_mesh = mp.solutions.face_mesh
# #
# #     skull_map = np.zeros((h, w), dtype=np.uint8)
# #
# #     with mp_face_mesh.FaceMesh(
# #             static_image_mode=True,
# #             max_num_faces=1,
# #             refine_landmarks=True,
# #             min_detection_confidence=0.5
# #     ) as face_mesh:
# #
# #         rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# #         results = face_mesh.process(rgb_image)
# #
# #         if results.multi_face_landmarks:
# #             print("[Info] Face landmarks detected successfully.")
# #             face_landmarks = results.multi_face_landmarks[0]
# #
# #             # Превращаем нормализованные координаты точек в пиксели
# #             points = []
# #             for landmark in face_landmarks.landmark:
# #                 px = int(landmark.x * w)
# #                 py = int(landmark.y * h)
# #                 points.append((px, py))
# #
# #             points = np.array(points, dtype=np.int32)
# #
# #             # Строим выпуклую оболочку (Convex Hull) по точкам лица
# #             hull = cv2.convexHull(points)
# #
# #             # Заливаем белым цветом (255) область головы на маске
# #             cv2.fillConvexPoly(skull_map, hull, 255)
# #
# #             # Опционально: делаем небольшое морфологическое расширение,
# #             # чтобы маска уверенно покрывала область роста волос
# #             kernel = np.ones((15, 15), np.uint8)
# #             skull_map = cv2.dilate(skull_map, kernel, iterations=1)
# #
# #         else:
# #             print("[Warning] No face detected in the image.")
# #
# #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
# #     cv2.imwrite(output_path, skull_map)
# #     print(f"[Success] Volumetric skull base generated: {output_path}")
# #
# #
# # if __name__ == "__main__":
# #     if len(sys.argv) < 3:
# #         print("Usage: python volumetric_skull_generator.py <input_image> <output_mask>")
# #         sys.exit(1)
# #
# #     weights_directory = os.environ.get("WEIGHTS_DIR", "/app/backend/apps/hairstyles/ai_engine/weights")
# #     generate_skull_base(sys.argv[1], sys.argv[2], weights_directory)
#
#
# import os
# import sys
# import cv2
# import numpy as np
# import mediapipe as mp
#
#
# def generate_skull_base(input_path: str, output_path: str, weights_dir: str):
#     print(f"[Info] Reading image from: {input_path}")
#     image = cv2.imread(input_path)
#     if image is None:
#         print(f"[Error] Failed to read input image: {input_path}")
#         sys.exit(1)
#
#     h, w, _ = image.shape
#     print(f"[Info] Image loaded successfully. Dimensions: {w}x{h}")
#
#     # Пути к конфигурации и весам 3DDFA_V2, которые мы скачали
#     config_path = os.path.join(weights_dir, "mb1_120x120.yml")
#     bfm_path = os.path.join(weights_dir, "bfm_noneck_v3.pkl")
#
#     if not os.path.exists(config_path) or not os.path.exists(bfm_path):
#         print(f"[Error] Missing model weights or configs in: {weights_dir}")
#         sys.exit(1)
#
#     print(f"[Info] Initializing 3DDFA_V2 model weights check passed in: {weights_dir}")
#
#     # Инициализируем MediaPipe Face Mesh для поиска точек лица внутри контейнера
#     mp_face_mesh = mp.solutions.face_mesh
#     skull_map = np.zeros((h, w), dtype=np.uint8)
#
#     with mp_face_mesh.FaceMesh(
#             static_image_mode=True,
#             max_num_faces=1,
#             refine_landmarks=True,
#             min_detection_confidence=0.5
#     ) as face_mesh:
#
#         rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(rgb_image)
#
#         if results.multi_face_landmarks:
#             print("[Info] Face landmarks detected successfully!")
#             face_landmarks = results.multi_face_landmarks[0]
#
#             points = []
#             for landmark in face_landmarks.landmark:
#                 px = int(landmark.x * w)
#                 py = int(landmark.y * h)
#                 points.append((px, py))
#
#             points = np.array(points, dtype=np.int32)
#             hull = cv2.convexHull(points)
#             cv2.fillConvexPoly(skull_map, hull, 255)
#
#             kernel = np.ones((15, 15), np.uint8)
#             skull_map = cv2.dilate(skull_map, kernel, iterations=1)
#         else:
#             print("[Warning] MediaPipe could not detect any face in this image!")
#
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     cv2.imwrite(output_path, skull_map)
#     print(f"[Success] Volumetric skull base generated: {output_path}")
#
#
# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         print("Usage: python volumetric_skull_generator.py <input_image> <output_mask>")
#         sys.exit(1)
#
#     weights_directory = os.environ.get("WEIGHTS_DIR", "/app/backend/apps/hairstyles/ai_engine/weights")
#     generate_skull_base(sys.argv[1], sys.argv[2], weights_directory)
#
#


# import os
# import sys
# import cv2
# import numpy as np
# import yaml
# import onnxruntime as ort
#
# # Импорты для работы с 3DDFA_V2
# try:
#     from utils.inference import parse_config, predict_dense
#     from utils.estimate_pose import estimate_pose
# except ImportError:
#     sys.path.append(os.path.join(os.path.dirname(__file__), "."))
#
#
# class VolumetricSkullGenerator:
#     def __init__(self, weights_dir: str):
#         self.weights_dir = weights_dir
#         self.config_path = os.path.join(weights_dir, "mb1_120x120.yml")
#         self.onnx_path = os.path.join(weights_dir, "mb1_120x120.onnx")
#
#         if not os.path.exists(self.onnx_path) or not os.path.exists(self.config_path):
#             raise FileNotFoundError(f"[Error] Missing weights/configs in: {weights_dir}")
#
#         with open(self.config_path, 'r') as f:
#             self.cfg = yaml.safe_load(f)
#
#         opts = ort.SessionOptions()
#         opts.intra_op_num_threads = 4  # Защита от зависания потоков на Mac M1 Max
#         self.ort_session = ort.InferenceSession(self.onnx_path, opts)
#
#     def _preprocess_from_hair_mask(self, image: np.ndarray, hair_mask: np.ndarray) -> tuple:
#         """
#         Вычисляет бокс лица и черепа на основе маски волос (без сторонних детекторов лиц).
#         """
#         h, w, _ = image.shape
#
#         # 1. Бинаризируем маску и инвертируем: лицо/кожа становятся белыми (255), волосы/фон — черными (0)
#         _, binary_mask = cv2.threshold(hair_mask, 127, 255, cv2.THRESH_BINARY_INV)
#
#         # 2. Ограничиваем область поиска верхней половиной кадра, чтобы шея/футболка не ломали бокс
#         search_zone = binary_mask[0:int(h * 0.70), :]
#
#         contours, _ = cv2.findContours(search_zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         if not contours:
#             print("[Docker Warning] Не удалось выделить овал по маске волос, fallback к центру кадра.")
#             return None, None
#
#         largest_contour = max(contours, key=cv2.contourArea)
#         fx, fy, fw, fh = cv2.boundingRect(largest_contour)
#
#         # 3. Формируем квадратный бокс с охватом макушки и лба под 3DDFA_V2
#         cx, cy = fx + fw // 2, fy + fh // 2
#         size = int(max(fw, fh) * 1.4)
#
#         x1 = max(0, cx - size // 2)
#         y1 = max(0, cy - size // 2)
#         x2 = min(w, cx + size // 2)
#         y2 = min(h, cy + size // 2)
#
#         # 4. Crop и нормализация под стандарт модели (120x120)
#         crop = image[y1:y2, x1:x2]
#         crop_resized = cv2.resize(crop, (120, 120), interpolation=cv2.INTER_LINEAR)
#
#         inp = crop_resized.astype(np.float32)
#         inp = (inp - 127.5) / 128.0
#         inp = inp.transpose(2, 0, 1)  # HWC в CHW
#         inp = np.expand_dims(inp, axis=0)  # CHW в NCHW
#
#         roi_box = [x1, y1, x2, y2]
#         return inp, roi_box
#
#     def generate_skull_base(self, input_path: str, output_path: str, bfm_path: str, hair_mask_path: str = None):
#         print(f"[Info] Reading image from: {input_path}")
#         image = cv2.imread(input_path)
#         if image is None:
#             print(f"[Error] Failed to read input image: {input_path}")
#             sys.exit(1)
#
#         h, w, _ = image.shape
#         inp_tensor, roi_box = None, None
#
#         # ШАГ 1: Пытаемся получить кроп через маску волос, если она передана
#         if hair_mask_path and os.path.exists(hair_mask_path):
#             print(f"[Info] Using hair mask for facial/cranial alignment: {hair_mask_path}")
#             hair_mask = cv2.imread(hair_mask_path, cv2.IMREAD_GRAYSCALE)
#             if hair_mask is not None:
#                 inp_tensor, roi_box = self._preprocess_from_hair_mask(image, hair_mask)
#
#         # Если маски нет или она пустая — аварийный фолбэк по центру
#         if inp_tensor is None:
#             print("[Info] Fallback to central crop estimation...")
#             size = min(h, w) // 2
#             x1, y1 = (w - size) // 2, (h - size) // 3
#             x2, y2 = x1 + size, y1 + size
#             roi_box = [x1, y1, x2, y2]
#
#             crop = image[y1:y2, x1:x2]
#             crop_resized = cv2.resize(crop, (120, 120), interpolation=cv2.INTER_LINEAR)
#             inp_tensor = crop_resized.astype(np.float32)
#             inp_tensor = (inp_tensor - 127.5) / 128.0
#             inp_tensor = inp_tensor.transpose(2, 0, 1)
#             inp_tensor = np.expand_dims(inp_tensor, axis=0)
#
#         # ШАГ 2: Инференс через ONNX
#         input_name = self.ort_session.get_inputs()[0].name
#         outputs = self.ort_session.run(None, {input_name: inp_tensor})
#         params = outputs[0].flatten()
#
#         # ШАГ 3: Реконструкция 3D сетки
#         try:
#             vertices = predict_dense(image, params, roi_box, bfm_path)
#         except Exception as e:
#             print(f"[Error] predict_dense failed: {e}")
#             raise e
#
#         # ШАГ 4: Отрисовка точной слитной маски (лицо + экстраполяция черепа)
#         skull_map = np.zeros((h, w), dtype=np.uint8)
#
#         if vertices is not None:
#             pts_2d = vertices[:2, :].T.astype(np.int32)
#
#             # Строим базовый контур лица по точкам сетки
#             hull = cv2.convexHull(pts_2d)
#             cv2.fillPoly(skull_map, [hull], 255)
#
#             # Точно вычисляем параметры черепа на основе реальной геометрии лица
#             top_y = int(np.min(pts_2d[:, 1]))           # Верхняя точка лба / бровей
#             center_x = int(np.mean(pts_2d[:, 0]))       # Центр лица по оси X
#             face_width = int(np.max(pts_2d[:, 0]) - np.min(pts_2d[:, 0]))  # Ширина лица
#
#             # Экстраполируем макушку вверх пропорционально ширине лица
#             dome_height = int(face_width * 0.75)
#
#             # Рисуем черепной купол, органично продолжающий форму головы выше лба
#             cv2.ellipse(
#                 skull_map,
#                 (center_x, top_y),
#                 (int(face_width * 0.52), dome_height),
#                 0, 180, 360, 255, -1
#             )
#
#             # Морфологическое сглаживание швов, чтобы переход между лицом и куполом был монолитным
#             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
#             skull_map = cv2.morphologyEx(skull_map, cv2.MORPH_CLOSE, kernel)
#
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         cv2.imwrite(output_path, skull_map)
#         print(f"[Success] Volumetric skull base generated: {output_path}")
#
#
# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         print("Usage: python volumetric_skull_generator.py <input_image> <output_mask> [hair_mask]")
#         sys.exit(1)
#
#     input_img_path = sys.argv[1]
#     output_mask_path = sys.argv[2]
#     hair_mask_path = sys.argv[3] if len(sys.argv) > 3 else None
#
#     weights_directory = os.environ.get("WEIGHTS_DIR", "/app/weights")
#     bfm_pickle_path = os.path.join(weights_directory, "bfm_noneck_v3.pkl")
#
#     generator = VolumetricSkullGenerator(weights_directory)
#     generator.generate_skull_base(input_img_path, output_mask_path, bfm_pickle_path, hair_mask_path)


import os
import sys
import subprocess
import yaml
import cv2
import numpy as np
import onnxruntime as ort
import pickle
from render_utils.render_ctypes import TrianglesMeshRender
from render_utils.TDDFA_ONNX import TDDFA_ONNX


class VolumetricSkullGenerator:
    def __init__(self, weights_dir: str):
        self.weights_dir = weights_dir
        self.config_path = os.path.join(weights_dir, "mb1_120x120.yml")
        self.onnx_path = os.path.join(weights_dir, "mb1_120x120.onnx")

        if not os.path.exists(self.onnx_path) or not os.path.exists(self.config_path):
            raise FileNotFoundError(f"[Error] Missing weights/configs in: {weights_dir}")

        with open(self.config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.ort_session = ort.InferenceSession(self.onnx_path, opts)

    def _ensure_docker_running(self):
        try:
            subprocess.run(["docker", "info"], capture_output=True, check=True)
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "[Error] Docker daemon is not running. "
                "Please make sure Docker Desktop is running in your system tray."
            )

    def _ensure_docker_image(self):
        print("[Info] Building Docker image for 3D Skull Generator (3DDFA_V2) with --no-cache...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        try:
            subprocess.run(
                [
                    "docker",
                    "build",
                    "--no-cache",
                    "-t",
                    "volumetric-skull-generator",
                    current_dir,
                ],
                check=True,
            )
            print("[Success] Docker image built successfully!")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"[Error] Failed to build Docker image: {e}")

    def _preprocess_from_hair_mask(self, image: np.ndarray, hair_mask: np.ndarray, debug_dir: str) -> tuple:
        h, w, _ = image.shape
        y_indices, x_indices = np.where(hair_mask > 127)

        if len(y_indices) == 0 or len(x_indices) == 0:
            print("[Debug] Маска волос пустая! Срабатывает fallback.")
            return None, None

        y_top = int(np.min(y_indices))
        x_left = int(np.min(x_indices))
        x_right = int(np.max(x_indices))

        print(f"[Debug] Анализ маски волос: макушка y={y_top}, ширина от x={x_left} до x={x_right}")

        face_w = x_right - x_left
        if face_w < 30:
            print("[Debug] Слишком узкая область волос, срабатывает fallback.")
            return None, None

        face_h = int(face_w * 1.45)

        x1 = max(0, x_left)
        y1 = max(0, y_top)
        x2 = min(w, x_right)
        y2 = min(h, y1 + face_h)

        roi_box = [x1, y1, x2, y2]
        print(f"[Debug] Рассчитанный ROI бокс по волосам: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        os.makedirs(debug_dir, exist_ok=True)

        debug_box_img = image.copy()
        cv2.rectangle(debug_box_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.imwrite(os.path.join(debug_dir, "debug_2_roi_box.png"), debug_box_img)

        crop = image[y1:y2, x1:x2]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            return None, None

        ch, cw, _ = crop.shape
        scale = 120.0 / max(ch, cw)
        resized_w = int(cw * scale)
        resized_h = int(ch * scale)
        resized_crop = cv2.resize(crop, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        crop_resized = np.zeros((120, 120, 3), dtype=np.uint8)
        start_x = (120 - resized_w) // 2
        start_y = (120 - resized_h) // 2
        crop_resized[start_y:start_y + resized_h, start_x:start_x + resized_w] = resized_crop

        cv2.imwrite(os.path.join(debug_dir, "debug_3_model_input.png"), crop_resized)

        inp = crop_resized.astype(np.float32)
        inp = (inp - 127.5) / 128.0
        inp = inp.transpose(2, 0, 1)
        inp = np.expand_dims(inp, axis=0)

        return inp, roi_box

    def generate_skull_base(self, input_path: str, output_path: str, bfm_path: str = None, hair_mask_path: str = None):
        print(f"[Info] Reading image from: {input_path}")
        image = cv2.imread(input_path)
        if image is None:
            print(f"[Error] Failed to read input image: {input_path}")
            sys.exit(1)

        h, w, _ = image.shape
        inp_tensor, roi_box = None, None

        debug_dir = os.path.join(os.path.dirname(output_path), "debug")
        os.makedirs(debug_dir, exist_ok=True)

        # ШАГ 1: Пытаемся использовать официальный детектор лиц и препроцессинг из репозитория
        try:
            try:
                from render_utils.FaceBoxes_ONNX import FaceBoxes_ONNX
            except ImportError:
                from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX

            face_boxes = FaceBoxes_ONNX()
            tddfa = TDDFA_ONNX(**self.cfg)

            boxes = face_boxes(image)
            if boxes is not None and len(boxes) > 0:
                print(f"[Info] Official FaceBoxes detected {len(boxes)} face(s).")
                param_lst, roi_box_lst = tddfa(image, boxes)
                if roi_box_lst is not None and len(roi_box_lst) > 0:
                    roi_box = roi_box_lst[0].astype(int).tolist()
                    x1, y1, x2, y2 = roi_box[0], roi_box[1], roi_box[2], roi_box[3]

                    debug_box_img = image.copy()
                    cv2.rectangle(debug_box_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    cv2.imwrite(os.path.join(debug_dir, "debug_2_roi_box_OFFICIAL.png"), debug_box_img)

                    crop = image[y1:y2, x1:x2]
                    if crop.size > 0:
                        ch, cw, _ = crop.shape
                        scale = 120.0 / max(ch, cw)
                        resized_w = int(cw * scale)
                        resized_h = int(ch * scale)
                        resized_crop = cv2.resize(crop, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

                        crop_resized = np.zeros((120, 120, 3), dtype=np.uint8)
                        start_x = (120 - resized_w) // 2
                        start_y = (120 - resized_h) // 2
                        crop_resized[start_y:start_y + resized_h, start_x:start_x + resized_w] = resized_crop

                        cv2.imwrite(os.path.join(debug_dir, "debug_3_model_input_OFFICIAL.png"), crop_resized)

                        inp_tensor = crop_resized.astype(np.float32)
                        inp_tensor = (inp_tensor - 127.5) / 128.0
                        inp_tensor = inp_tensor.transpose(2, 0, 1)
                        inp_tensor = np.expand_dims(inp_tensor, axis=0)
        except Exception as e:
            print(f"[Warning] Official FaceBoxes/TDDFA method skipped or failed: {e}")

        # ШАГ 2: Если официальный метод не сработал, пробуем твой метод по маске волос
        if inp_tensor is None and hair_mask_path and os.path.exists(hair_mask_path):
            print(f"[Info] Fallback to hair mask: {hair_mask_path}")
            hair_mask = cv2.imread(hair_mask_path, cv2.IMREAD_GRAYSCALE)
            if hair_mask is not None:
                inp_tensor, roi_box = self._preprocess_from_hair_mask(image, hair_mask, debug_dir)

        # ШАГ 3: Фолбэк на центральный кроп, если ничего другое не сработало
        if inp_tensor is None:
            print("[Info] Fallback to central crop estimation...")
            size = min(h, w) // 2
            x1, y1 = (w - size) // 2, (h - size) // 3
            x2, y2 = x1 + size, y1 + size
            roi_box = [x1, y1, x2, y2]

            debug_box_img = image.copy()
            cv2.rectangle(debug_box_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.imwrite(os.path.join(debug_dir, "debug_2_roi_box_FALLBACK.png"), debug_box_img)

            crop = image[y1:y2, x1:x2]
            ch, cw, _ = crop.shape
            scale = 120.0 / max(ch, cw)
            resized_w = int(cw * scale)
            resized_h = int(ch * scale)
            resized_crop = cv2.resize(crop, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

            crop_resized = np.zeros((120, 120, 3), dtype=np.uint8)
            start_x = (120 - resized_w) // 2
            start_y = (120 - resized_h) // 2
            crop_resized[start_y:start_y + resized_h, start_x:start_x + resized_w] = resized_crop

            cv2.imwrite(os.path.join(debug_dir, "debug_3_model_input_FALLBACK.png"), crop_resized)

            inp_tensor = crop_resized.astype(np.float32)
            inp_tensor = (inp_tensor - 127.5) / 128.0
            inp_tensor = inp_tensor.transpose(2, 0, 1)
            inp_tensor = np.expand_dims(inp_tensor, axis=0)

        # ШАГ 4: Инференс модели и получение плотной карты позиций
        input_name = self.ort_session.get_inputs()[0].name
        outputs = self.ort_session.run(None, {input_name: inp_tensor})

        pred_map = outputs[0]
        if len(pred_map.shape) == 4:
            pred_map = np.squeeze(pred_map, axis=0)

        print(f"[Info] ONNX inference successful, dense map shape: {pred_map.shape}")

        # ШАГ 5: Рендеринг точной 3D-модели лица через нативный C-рендерер и достройка купола
        skull_map = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = roi_box
        box_w = x2 - x1
        box_h = y2 - y1

        so_path = os.path.join(os.path.dirname(__file__), "render_utils", "render.so")
        pkl_path = bfm_path or os.path.join(os.path.dirname(__file__), "render_utils", "bfm_noneck_v3.pkl")

        if os.path.exists(so_path) and os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    bfm_data = pickle.load(f)
                triangles = bfm_data.get("tri")

                vertices = pred_map[:3, :, :].reshape(3, -1).T
                vertices[:, 0] = vertices[:, 0] * (box_w / 120.0) + x1
                vertices[:, 1] = vertices[:, 1] * (box_h / 120.0) + y1

                mesh_renderer = TrianglesMeshRender(clibs=so_path)
                overlay_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                ver_ct = np.ascontiguousarray(vertices.T)
                mesh_renderer(ver_ct, triangles, bg=overlay_canvas)

                gray_face = cv2.cvtColor(overlay_canvas, cv2.COLOR_BGR2GRAY)
                _, skull_map = cv2.threshold(gray_face, 10, 255, cv2.THRESH_BINARY)
                print("[Success] Native C-renderer face mask generated successfully.")
            except Exception as e:
                import traceback
                print(f"[Warning] Native C-render failed, details:")
                traceback.print_exc()

        # Фолбэк на случай сбоя рендерера
        if np.count_nonzero(skull_map) < 100:
            print("[Warning] Fallback anatomical shape applied...")
            cx, cy = (x1 + x2) // 2, y1 + box_h // 2
            cv2.ellipse(skull_map, (cx, cy), (box_w // 2, box_h // 2), 0, 0, 360, 255, -1)

        # Достраиваем верхний купол черепа под прическу
        y_indices, x_indices = np.where(skull_map > 0)
        if len(y_indices) > 0:
            face_top_y = y_indices.min()
            cv2.circle(skull_map, ((x1 + x2) // 2, max(y1, face_top_y - int(box_h * 0.15))), int(box_w * 0.4), 255, -1)

        # Финальное сглаживание купола
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        skull_map = cv2.morphologyEx(skull_map, cv2.MORPH_CLOSE, kernel)
        skull_map = cv2.GaussianBlur(skull_map, (41, 41), 0)
        _, skull_map = cv2.threshold(skull_map, 127, 255, cv2.THRESH_BINARY)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, skull_map)
        print(f"[Success] Volumetric skull base generated securely: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python volumetric_skull_generator.py <input_image> <output_mask> [hair_mask]")
        sys.exit(1)

    input_img_path = sys.argv[1]
    output_mask_path = sys.argv[2]
    hair_mask_path = sys.argv[3] if len(sys.argv) > 3 else None

    weights_directory = os.environ.get("WEIGHTS_DIR", "/app/weights")
    bfm_pickle_path = os.path.join(weights_directory, "bfm_noneck_v3.pkl")

    generator = VolumetricSkullGenerator(weights_directory)
    generator.generate_skull_base(input_img_path, output_mask_path, bfm_pickle_path, hair_mask_path)
