import os
import yaml
import cv2
import numpy as np

class FaceExtractor:
    def __init__(self, weights_dir: str):
        self.weights_dir = weights_dir
        self.config_path = os.path.join(weights_dir, "mb1_120x120.yml")
        self.onnx_path = os.path.join(weights_dir, "mb1_120x120.onnx")

        if not os.path.exists(self.onnx_path) or not os.path.exists(self.config_path):
            raise FileNotFoundError(f"[Error] Missing weights/configs in: {weights_dir}")

        with open(self.config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

    def extract_face_features(self, input_path: str, output_path: str):
        """
        Main method: detects face, extracts dense facial vertices,
        rotation matrix, and saves debug images for visual inspection.
        """
        print(f"[Info] Reading image from: {input_path}")
        image = cv2.imread(input_path)
        if image is None:
            print(f"[Error] Failed to read input image: {input_path}")
            return None

        h, w, _ = image.shape
        debug_dir = os.path.join(os.path.dirname(output_path), "debug")
        os.makedirs(debug_dir, exist_ok=True)

        from render_utils.FaceBoxes_ONNX import FaceBoxes_ONNX
        from render_utils.TDDFA_ONNX import TDDFA_ONNX
        from render_utils.tddfa_util import _parse_param

        # 1. Face detection via FaceBoxes
        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**self.cfg)

        boxes = face_boxes(image)
        if boxes is None or len(boxes) == 0:
            print('[Error] No face detected, exit')
            return None

        boxes = np.array(boxes, dtype=np.float32)
        param_lst, roi_box_lst = tddfa(image, boxes)

        if not param_lst or len(param_lst) == 0 or not roi_box_lst or len(roi_box_lst) == 0:
            raise RuntimeError("[Error] TDDFA failed to generate dense parameters for the face.")

        # Save ROI debug visualization
        roi_box = np.array(roi_box_lst[0], dtype=np.float32).astype(int).tolist()
        x1, y1, x2, y2 = max(roi_box[0], 0), max(roi_box[1], 0), min(roi_box[2], w - 1), min(roi_box[3], h - 1)

        debug_box_img = image.copy()
        cv2.rectangle(debug_box_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.imwrite(os.path.join(debug_dir, "debug_2_roi_box_OFFICIAL.png"), debug_box_img)

        # 2. Dense facial geometry reconstruction
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
        ver_face = ver_lst[0].astype(np.float32)  # Facial shape (3, N)
        # 2.1. Sparse keypoints reconstruction (ключевые точки лица)
        ver_sparse_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
        ver_landmarks = ver_sparse_lst[0].astype(np.float32)

        # 3. Extract rotation matrix R and Euler angles
        param = param_lst[0]
        R, offset, _, _ = _parse_param(param)

        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        if sy >= 1e-6:
            pitch = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[1, 0], R[0, 0])
        else:
            pitch = np.arctan2(-R[1, 2], R[1, 1])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = 0

        print(
            f"[Info] Euler angles (deg) -> Pitch: {np.degrees(pitch):.1f}, Yaw: {np.degrees(yaw):.1f}, Roll: {np.degrees(roll):.1f}")

        # 4. Save debug face projection overlay
        try:
            debug_overlay = image.copy()
            pts2d_all = ver_face[:2, :].T.astype(np.int32)
            for pt in pts2d_all[::15]:
                cv2.circle(debug_overlay, (pt[0], pt[1]), 1, (0, 255, 0), -1)
            cv2.imwrite(os.path.join(debug_dir, "debug_face_overlay.png"), debug_overlay)

        except Exception as e:
             print(f"[Warning] Failed to save debug overlay: {e}")
        return {
            "ver_face": ver_face,  # Dense facial vertices (3, N)
            "rotation": R,          # 3x3 rotation matrix
            "ver_landmarks": ver_landmarks,#key facial points
        }


if __name__ == "__main__":
    import sys
    import io
    import base64
    import numpy as np

    if len(sys.argv) < 2:
        print("Usage: python face_feature_extractor.py <input_image>", file=sys.stderr)
        sys.exit(1)

    input_img_path = sys.argv[1]
    weights_directory = os.environ.get("WEIGHTS_DIR", "/app/weights")

    generator = FaceExtractor(weights_directory)

    # Dummy path for compatibility since we don't save masks to disk anymore
    dummy_output = "/tmp/debug_output.png"
    result_dict = generator.extract_face_features(input_img_path, dummy_output)

    if result_dict is not None and "ver_face" in result_dict and "rotation" in result_dict:
        ver_face = result_dict["ver_face"]
        ver_landmarks = result_dict.get("ver_landmarks")
        R = result_dict["rotation"]

        # 1. Create an in-memory bytes buffer
        buffer = io.BytesIO()

        # 2. Save arrays into the buffer with compression
        np.savez_compressed(buffer, ver_face=ver_face, ver_landmarks=ver_landmarks, rotation=R)

        # 3. Convert binary archive to a safe Base64 string
        buffer.seek(0)
        b64_string = base64.b64encode(buffer.read()).decode('utf-8')

        # 4. Write clean delimited string to stdout
        sys.stdout.write(f"===DATA_START==={b64_string}===DATA_END===\n")
        sys.stdout.flush()
        sys.exit(0)
    else:
        print("[Error] Feature extraction returned None or invalid structure.", file=sys.stderr)
        sys.exit(1)

