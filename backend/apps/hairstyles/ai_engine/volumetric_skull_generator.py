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


class VolumetricSkullGenerator:
    def __init__(self, weights_dir: str):
        self.weights_dir = weights_dir
        self.config_path = os.path.join(weights_dir, "mb1_120x120.yml")
        self.onnx_path = os.path.join(weights_dir, "mb1_120x120.onnx")

        if not os.path.exists(self.onnx_path) or not os.path.exists(self.config_path):
            raise FileNotFoundError(f"[Error] Missing weights/configs in: {weights_dir}")

        with open(self.config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

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
        roi_box = None

        debug_dir = os.path.join(os.path.dirname(output_path), "debug")
        os.makedirs(debug_dir, exist_ok=True)

        from render_utils.FaceBoxes_ONNX import FaceBoxes_ONNX
        from render_utils.TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**self.cfg)

        # 1. Детекция лиц через официальный FaceBoxes
        boxes = face_boxes(image)
        if boxes is None or len(boxes) == 0:
            print('[Error] No face detected, exit')
            return False

        print(f"[Info] Official FaceBoxes detected {len(boxes)} face(s).")
        boxes = np.array(boxes, dtype=np.float32)

        # 2. Получение параметров регрессии через официальный TDDFA
        param_lst, roi_box_lst = tddfa(image, boxes)

        if not param_lst or len(param_lst) == 0 or not roi_box_lst or len(roi_box_lst) == 0:
            raise RuntimeError("[Error] TDDFA failed to generate dense parameters for the face.")

        # Берем первый найденный бокс лица
        roi_box = np.array(roi_box_lst[0], dtype=np.float32).astype(int).tolist()
        x1, y1, x2, y2 = roi_box[0], roi_box[1], roi_box[2], roi_box[3]

        # Сохраняем дебаг-картинку с найденным ROI (ваша логика отладки)
        debug_box_img = image.copy()
        cv2.rectangle(debug_box_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.imwrite(os.path.join(debug_dir, "debug_2_roi_box_OFFICIAL.png"), debug_box_img)

        # # 3. Реконструкция плотных вертексов через официальный метод TDDFA
        # dense_vertices_list = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
        #
        # # 4. Рендеринг 3D-модели лица через нативный C-рендерер
        # skull_map = np.zeros((h, w), dtype=np.uint8)
        # box_w = x2 - x1
        # box_h = y2 - y1
        #
        # so_path = os.path.join(os.path.dirname(__file__), "render_utils", "asset", "render.so")
        # if not os.path.exists(so_path):
        #     so_path = os.path.join(os.path.dirname(__file__), "render_utils", "render.so")
        #
        # pkl_path = bfm_path or os.path.join(os.path.dirname(__file__), "render_utils", "bfm_noneck_v3.pkl")
        #
        # if os.path.exists(so_path) and os.path.exists(pkl_path):
        #     try:
        #         with open(pkl_path, "rb") as f:
        #             bfm_data = pickle.load(f)
        #         triangles = bfm_data.get("tri")
        #
        #         overlay_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        #         from render_utils.render_ctypes import render_app
        #
        #         for ver_ in dense_vertices_list:
        #             ver = np.ascontiguousarray(ver_.T)
        #             render_app(ver, triangles, bg=overlay_canvas)
        #
        #         gray_face = cv2.cvtColor(overlay_canvas, cv2.COLOR_BGR2GRAY)
        #         _, skull_map = cv2.threshold(gray_face, 10, 255, cv2.THRESH_BINARY)
        #         print("[Success] Native C-renderer face mask generated successfully.")
        #     except Exception as e:
        #         import traceback
        #         traceback.print_exc()
        #
        # # Фолбэк, если рендерер не сработал
        # if np.count_nonzero(skull_map) < 100:
        #     print("[Warning] Fallback to ROI box mask...")
        #     cv2.rectangle(skull_map, (x1, y1), (x2, y2), 255, -1)
        #
        # # 5. Достройка верхнего купола черепа под прическу (ваша логика)
        # y_indices, x_indices = np.where(skull_map > 0)
        # if len(y_indices) > 0:
        #     face_top_y = y_indices.min()
        #     cv2.circle(skull_map, ((x1 + x2) // 2, max(y1, face_top_y - int(box_h * 0.15))), int(box_w * 0.4), 255, -1)
        #
        # # Финальное сглаживание и морфология купола
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        # skull_map = cv2.morphologyEx(skull_map, cv2.MORPH_CLOSE, kernel)
        # skull_map = cv2.GaussianBlur(skull_map, (41, 41), 0)
        # _, skull_map = cv2.threshold(skull_map, 127, 255, cv2.THRESH_BINARY)
        #
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # cv2.imwrite(output_path, skull_map)
        # print(f"[Success] Skull base mask generated: {output_path}")
        # return True
        # =====================================================================
        # 3. Реконструкция плотных вертексов лица
        # =====================================================================
        # Явно передаем dense_flag=True для получения полной маски лица (~40 000 точек)
        dense_vertices_list = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)

        # =====================================================================
        # 4. Рендеринг анатомической объемной маски лица через карту глубины
        # =====================================================================
        skull_map = np.zeros((h, w), dtype=np.uint8)

        if dense_vertices_list:
            try:
                # Извлекаем первую матрицу 3D вертексов (3, N)
                ver = dense_vertices_list[0]

                # Координаты X и Y проецируем на плоскость картинки
                pts2d = ver[:2, :].T.astype(np.int32)

                # Координата Z отвечает за глубину (объем)
                z = ver[2, :]

                # Нормализуем Z в диапазон от 150 до 255 (чтобы лицо было объемным, но ярким)
                z_min, z_max = np.min(z), np.max(z)
                if z_max - z_min > 0:
                    z_norm = 150 + ((z - z_min) / (z_max - z_min) * 105)
                else:
                    z_norm = np.ones_like(z) * 255
                z_norm = z_norm.astype(np.uint8)

                # Создаем временный холст для карты глубины
                depth_canvas = np.zeros((h, w), dtype=np.uint8)

                # Ограничиваем координаты, чтобы не выйти за границы картинки
                valid_idx = (pts2d[:, 0] >= 0) & (pts2d[:, 0] < w) & (pts2d[:, 1] >= 0) & (pts2d[:, 1] < h)
                x_v = pts2d[valid_idx, 0]
                y_v = pts2d[valid_idx, 1]
                z_v = z_norm[valid_idx]

                # Быстро записываем точки объема на холст без циклов Python
                depth_canvas[y_v, x_v] = z_v

                # Строим плотную выпуклую оболочку лица, чтобы залить пустоты между точками
                hull = cv2.convexHull(pts2d)
                mask_hull = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask_hull, [hull], 255)

                # Размываем точки для получения гладкого рельефа кожи (носа, скул)
                blurred_depth = cv2.GaussianBlur(depth_canvas, (15, 15), 0)

                # Применяем маску лица, чтобы объем не размывался на черный фон
                skull_map = cv2.bitwise_and(blurred_depth, blurred_depth, mask=mask_hull)

                # Переводим логическую маску в uint8 формат
                empty_pixels_mask = (skull_map == 0).astype(np.uint8) * 255

                # Выравниваем базовую яркость подложки лица (подбородок и щеки)
                fallback_background = cv2.bitwise_and(mask_hull, mask_hull, mask=empty_pixels_mask)

                # ИСПРАВЛЕНИЕ ОШИБКИ NUMPY: используем np.where для безопасного объединения рельефа и подложки
                bg_intensity = (fallback_background.astype(np.float32) * 150 / 255).astype(np.uint8)
                skull_map = np.where(skull_map == 0, bg_intensity, skull_map).astype(np.uint8)

                # Финальное ограничение по маске лица
                skull_map = cv2.bitwise_and(skull_map, mask_hull)

                # ИСПРАВЛЕНИЕ NUMPY: используем np.where для безопасного объединения рельефа и подложки
                bg_intensity = (fallback_background.astype(np.float32) * 150 / 255).astype(np.uint8)
                skull_map = np.where(skull_map == 0, bg_intensity, skull_map).astype(np.uint8)

                # ─── ДОБАВЛЯЕМ ЭТОТ БЛОК ДЛЯ ПРОЯВЛЕНИЯ ТЕНЕЙ ───────────────────
                # Растягиваем диапазон яркости только внутри маски лица, чтобы проявить рельеф
                active_pixels = skull_map[mask_hull > 0]
                if len(active_pixels) > 0:
                    s_min, s_max = np.min(active_pixels), np.max(active_pixels)
                    if s_max - s_min > 0:
                        # Картографируем глубину от 80 (темно-серый для впадин) до 255 (белый для носа)
                        skull_map = np.where(
                            mask_hull > 0,
                            (80 + ((skull_map.astype(np.float32) - s_min) / (s_max - s_min) * 175)).astype(np.uint8),
                            0
                        )
                # ────────────────────────────────────────────────────────────────

                # Финальное ограничение по маске лица
                skull_map = cv2.bitwise_and(skull_map, mask_hull)

                print("[Success] Volumetric 3D depth face surface generated successfully via Vectorization!")


            except Exception as e:
                print(f"[Error in Depth rendering]: {e}")
                # Надежный плоский фолбэк по контуру, если математика подвела
                pts2d = dense_vertices_list[0][:2, :].T.astype(np.int32)
                hull = cv2.convexHull(pts2d)
                cv2.fillPoly(skull_map, [hull], 255)

        # =====================================================================
        # 5. Достройка объемного купола черепа строго под анатомию
        # =====================================================================
        y_indices, x_indices = np.where(skull_map > 0)
        if len(y_indices) > 0 and len(x_indices) > 0:
            face_top_y = y_indices.min()
            face_left_x = x_indices.min()
            face_right_x = x_indices.max()

            center_x = (face_left_x + face_right_x) // 2
            face_width = face_right_x - face_left_x

            # Радиус черепа
            circle_radius = int(face_width * 0.44)
            circle_center_y = int(face_top_y + (circle_radius * 0.35))

            # Создаем объемную полусферу (градиент от центра к краям)
            dome_mask = np.zeros((h, w), dtype=np.uint8)
            for r in range(circle_radius, 0, -1):
                # Чем ближе к центру сферы, тем ярче (белее), имитируя объемный затылок
                brightness = int(200 + (55 * (1.0 - r / circle_radius)))
                cv2.circle(dome_mask, (center_x, circle_center_y), r, brightness, -1)

            # Объединяем объемное лицо и объемный купол черепа (берем максимальную яркость точек)
            skull_map = cv2.max(skull_map, dome_mask)
            print("[Success] 3D volumetric skull dome blended perfectly.")


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
