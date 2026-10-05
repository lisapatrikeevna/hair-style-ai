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

        # =====================================================================
        # 3. Реконструкция плотных вертексов лица
        # =====================================================================
        # Явно передаем dense_flag=True для получения полной маски лица (~40 000 точек)
        # Здесь ты получаешь реальные 3D-координаты лица (X, Y, Z)
        dense_vertices_list = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
        print('dense_vertices_list',dense_vertices_list)
        ver_face = dense_vertices_list[0]  # Формат: [3, N]
        print('ver_face',ver_face)

        # Индексы ключевых антропологических точек в стандартной топологии BFM
        IDX_LEFT_CHEEK = 1841
        IDX_RIGHT_CHEEK = 4307
        IDX_NASION = 8241
        IDX_CHIN = 8215

        pt_left = ver_face[:, IDX_LEFT_CHEEK]
        pt_right = ver_face[:, IDX_RIGHT_CHEEK]
        pt_nasion = ver_face[:, IDX_NASION]
        pt_chin = ver_face[:, IDX_CHIN]

        # ---- АНТРОПОЛОГИЧЕСКИЙ РАСЧЕТ СТРОПИЛЬНОЙ СИСТЕМЫ ----
        face_width = np.linalg.norm(pt_left - pt_right)
        face_height = np.linalg.norm(pt_nasion - pt_chin)
        print('face_width:',face_width,'face_height:',face_height)

        # Вычисляем анатомический центр для построения геометрии купола
        center_x = (pt_left[0] + pt_right[0]) / 2.0
        center_y = pt_nasion[1]
        center_z = pt_nasion[2]

        # Краниометрические радиусы купола черепа
        r_x = (face_width * 0.94) / 2.0
        r_y = face_height * 0.88  # Слегка увеличили для естественной высоты темени
        r_z = face_width * 1.25

        # ---- МАТЕМАТИЧЕСКАЯ ГЕНЕРАЦИЯ 3D КУПОЛА ЧЕРЕПА ----
        skull_vertices = []
        u_steps = 30
        v_steps = 30

        for i in range(u_steps):
            v_angle = (i / (u_steps - 1)) * (np.pi / 2.0)  # Заднее полушарие
            for j in range(v_steps):
                u_angle = (j / (v_steps - 1)) * np.pi  # Дуга слева направо

                x = center_x + r_x * np.cos(u_angle) * np.sin(v_angle)
                # ЗНАК МИНУС: Переворачиваем Y вверх (в 3D пространстве макушка растет вверх)
                y = center_y - r_y * np.sin(u_angle) * np.sin(v_angle)
                # ЗНАК ПЛЮС: Уводим затылок строго назад по оси Z, углубляя череп
                z = center_z + r_z * np.cos(v_angle)

                skull_vertices.append([x, y, z])

        skull_vertices = np.array(skull_vertices).T  # [3, M]
        full_vertices = np.hstack([ver_face, skull_vertices])  # [3, N + M]
        print(f"[DEBUG GEOMETRY] Skull vertices shape: {skull_vertices.shape}")
        print(
            f"[DEBUG GEOMETRY] Skull Z-axis values (min/max): {np.min(skull_vertices[2, :])} / {np.max(skull_vertices[2, :])}")

        # ---- ГЛОБАЛЬНОЕ ЦЕНТРИРОВАНИЕ И МАСШТАБИРОВАНИЕ ДЛЯ 3D ВЬЮЕРА ----
        # Вычисляем общий геометрический центр получившейся модели
        model_centerOLD = np.mean(full_vertices, axis=1, keepdims=True)
        model_center = pt_nasion.reshape(3, 1)
        print('model_center:',model_center,'model_centerOLD:',model_centerOLD)
        # Сдвигаем модель в 0,0,0 и уменьшаем масштаб в 1000 раз (переводим пиксели в метры)
        normalized_vertices = (full_vertices - model_center) / 1000.0

        # Инвертируем ось Y для всей финальной модели, так как в веб-просмотрщиках верх — это +Y
        normalized_vertices[1, :] = -normalized_vertices[1, :]
        print('normalized_vertices',normalized_vertices)

        # ---- ЗАПИСЬ В OBJ С ФИЛЬТРАЦИЕЙ БИТЫХ ИНДЕКСОВ ----
        obj_output_path = output_path.replace(".png", ".obj")
        print(f"[Info] Saving complete normalized 3D skull mesh to: {obj_output_path}")

        try:
            pkl_path = bfm_path or os.path.join(os.path.dirname(__file__), "render_utils", "bfm_noneck_v3.pkl")
            num_face_vertices = ver_face.shape[1]  # Реальное количество вершин лица (38365)
            print('num_face_vertices',num_face_vertices)

            with open(obj_output_path, "w") as obj_file:
                # Записываем нормализованные вершины (Лицо + Затылок)
                for v in normalized_vertices.T:
                    obj_file.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

                # Безопасно загружаем и фильтруем треугольники
                if os.path.exists(pkl_path):
                    with open(pkl_path, "rb") as f:
                        bfm_data = pickle.load(f)
                    triangles = bfm_data.get("tri")  # shape: (3, num_triangles)

                    # Записываем ТОЛЬКО те треугольники, индексы которых лежат внутри массива лица
                    total_written_vertices = full_vertices.shape[1]  # Это 38365 + 900 = 39265
                    print('total_written_vertices',total_written_vertices)

                    # triangles имеет shape (3, N) — каждый столбец это один треугольник.
                    # Итерируемся по .T (транспонированному), чтобы проходить по столбцам, а не по строкам.
                    written_count = 0
                    for t in triangles.T:
                        if t[0] < num_face_vertices and t[1] < num_face_vertices and t[2] < num_face_vertices:
                            obj_file.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
                            written_count += 1
                    print(f"[DEBUG OBJ] Written {written_count} face triangles out of {triangles.shape[1]} total")

            print(f"[Success] 3D OBJ model is optimized and ready for web viewer! Path: {obj_output_path}")
        except Exception as obj_err:
            print(f"[Error] Failed to write OBJ mesh: {obj_err}")


        # =====================================================================
        # 4. Рендеринг анатомического 3D-лица через родной Си-рендерер (render.c)
        # =====================================================================
        # А здесь ты создаешь ПЛОСКУЮ 2D-картинку (черно-белую матрицу пикселей)
        skull_map = np.zeros((h, w), dtype=np.uint8)

        so_path = os.path.join(os.path.dirname(__file__), "render_utils", "asset", "render.so")
        if not os.path.exists(so_path):
            so_path = os.path.join(os.path.dirname(__file__), "render_utils", "render.so")

        # Берем оригинальный путь к pkl-файлу для извлечения топологии треугольников
        pkl_path = bfm_path or os.path.join(os.path.dirname(__file__), "render_utils", "bfm_noneck_v3.pkl")

        if os.path.exists(so_path) and os.path.exists(pkl_path):
            try:
                # Загружаем треугольники лица из pkl
                with open(pkl_path, "rb") as f:
                    bfm_data = pickle.load(f)
                triangles = bfm_data.get("tri")

                # Создаем чистый холст, на котором Си-код будет рисовать маску
                overlay_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                from render_utils.render_ctypes import render_app

                for ver_ in dense_vertices_list:
                    # Транспонируем матрицу и принудительно переводим её в float32!
                    # Теперь Си-код прочитает координаты без смещения памяти.
                    ver = np.ascontiguousarray(ver_.T, dtype=np.float32)
                    num_ver = ver.shape[0]  # Количество вершин в этом массиве (38365)

                    # КРИТИЧЕСКИ ВАЖНО: Фильтруем треугольники, чтобы все индексы были < num_ver.
                    # BFM содержит треугольники с индексами до 46851, но вершин только 38365.
                    # Без фильтрации Си-код выходит за пределы массива вершин,
                    # что приводит к повреждению кучи и краху free(): invalid next size.
                    valid_mask = (triangles[0, :] < num_ver) & (triangles[1, :] < num_ver) & (triangles[2, :] < num_ver)
                    tri_filtered = triangles[:, valid_mask]  # shape: (3, valid_count)

                    # Принудительно переводим треугольники в int32 под Си-указатели
                    tri = np.ascontiguousarray(tri_filtered, dtype=np.int32)

                    # ─── РАССТАВЛЯЕМ ПРИНТЫ ДЛЯ ПРОВЕРКИ ПАМЯТИ ──────────────────
                    print(f"[DEBUG C-RENDER] ver shape: {ver.shape}, dtype: {ver.dtype}")
                    print(f"[DEBUG C-RENDER] tri shape: {tri.shape}, dtype: {tri.dtype}")
                    print(f"[DEBUG C-RENDER] tri min index: {tri.min()}, max index: {tri.max()}")
                    print(f"[DEBUG C-RENDER] Filtered out {triangles.shape[1] - tri_filtered.shape[1]} out-of-bounds triangles")
                    print(f"[DEBUG C-RENDER] canvas shape: {overlay_canvas.shape}, non-zero pixels before: {np.count_nonzero(overlay_canvas)}")
                    # ────────────────────────────────────────────────────────────


                    # Вызываем Си-функцию, которая закрасит маску лица
                    render_app(ver, tri, bg=overlay_canvas)

                # Превращаем цветной трехмерный Си-рельеф в маску
                gray_face = cv2.cvtColor(overlay_canvas, cv2.COLOR_BGR2GRAY)
                _, skull_map = cv2.threshold(gray_face, 10, 255, cv2.THRESH_BINARY)
                print("[Success] Native C-renderer face mask generated anatomically via render.c!")
            except Exception as e:
                print(f"[Warning] Native C-renderer failed: {e}. Using landmarks fallback.")
                # Надежный плоский фолбэк по контуру, если Си-библиотека выдала ошибку
                pts2d = dense_vertices_list[0][:2, :].T.astype(np.int32)
                hull = cv2.convexHull(pts2d)
                cv2.fillPoly(skull_map, [hull], 255)

        # =====================================================================
        # 5. Достройка верхнего купола черепа строго под анатомию вашего лица
        # =====================================================================
        y_indices, x_indices = np.where(skull_map > 0)
        if len(y_indices) > 0 and len(x_indices) > 0:
            face_top_y = y_indices.min()
            face_left_x = x_indices.min()
            face_right_x = x_indices.max()

            # Динамически вычисляем центр головы и её реальную ширину на фото
            center_x = (face_left_x + face_right_x) // 2
            face_width = face_right_x - face_left_x

            # Строим верхний анатомический купол головы под прическу
            circle_radius = int(face_width * 0.46)
            circle_center_y = max(y1, face_top_y - int(circle_radius * 0.1))

            cv2.circle(skull_map, (center_x, circle_center_y), circle_radius, 255, -1)

        # Финальное сглаживание швов для плавного силуэта головы
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        skull_map = cv2.morphologyEx(skull_map, cv2.MORPH_CLOSE, kernel)
        skull_map = cv2.GaussianBlur(skull_map, (21, 21), 0)
        _, skull_map = cv2.threshold(skull_map, 127, 255, cv2.THRESH_BINARY)

        # Записываем результат на диск
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, skull_map)
        print(f"[Success] Beautiful anatomical skull mask generated: {output_path}")
        # return True

        # СУПЕР-ДЕБАГ: Накладываем маску на оригинальное фото, чтобы увидеть анатомию
        try:
            # Создаем цветную версию маски (RGB)
            overlay_mask = cv2.merge([skull_map, skull_map, skull_map])
            # Смешиваем оригинальное фото и маску 50/50
            debug_blend = cv2.addWeighted(image, 0.6, overlay_mask, 0.4, 0)

            # Сохраняем в папку дебага
            debug_blend_path = os.path.join(debug_dir, "debug_5_anatomical_blend.png")
            cv2.imwrite(debug_blend_path, debug_blend)
            print(f"[Success] Debug anatomical overlay generated: {debug_blend_path}")
        except Exception as overlay_err:
            print(f"[Warning] Failed to generate debug blend: {overlay_err}")

        return True


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


