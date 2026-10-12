import os
import subprocess
import json
import numpy as np
import io
import base64
import re


class FaceGeometryBuilder:

    def __init__(self):
        self._ensure_docker_running()
        self._ensure_docker_image()

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
        image_name = "face_feature_extractor"

        # Check if the image already exists locally to avoid 200+ seconds rebuilds
        inspect_process = subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True,
            text=True
        )

        if inspect_process.returncode == 0:
            print(f"[Info] Docker image '{image_name}' already exists. Skipping build.")
            return

        print(f"[Info] Building Docker image for 3D Skull Generator (3DDFA_V2)...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        try:
            subprocess.run(
                [
                    "docker",
                    "build",
                    "-t",
                    image_name,
                    current_dir,
                ],
                check=True,
            )
            print("[Success] Docker image built successfully!")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"[Error] Failed to build Docker image: {e}")

    def clean_up_docker(self):
        """
        Safe cleanup: removes stopped containers and dangling images
        without destroying the build cache of our main image.
        """
        print("[Info] Cleaning up unused Docker resources...")
        try:
            # Remove stopped containers
            subprocess.run(["docker", "container", "prune", "-f"], capture_output=True)
            # Remove dangling (untagged) images to free up space safely
            subprocess.run(["docker", "image", "prune", "-f"], capture_output=True)
            print("[Success] Docker cleanup completed!")
        except Exception as e:
            print(f"[Warning] Failed to clean up Docker: {e}")

    def build_bald_dome(self, image_path: str, output_path: str) -> dict:
        abs_input = os.path.abspath(image_path)
        abs_output = os.path.abspath(output_path)

        input_dir = os.path.dirname(abs_input)
        output_dir = os.path.dirname(abs_output)

        input_name = os.path.basename(abs_input)
        output_name = os.path.basename(abs_output)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(current_dir, "weights", "skull_weights")

        cmd = [
            "docker",
            "run",
            "--rm",
            "-v", f"{input_dir}:/data_in",
            "-v", f"{output_dir}:/data_out",
            "-v", f"{weights_dir}:/app/weights",
            "-e", "WEIGHTS_DIR=/app/weights",
            "face_feature_extractor",
            f"/data_in/{input_name}",
            f"/data_out/{output_name}",
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )

            if result.stderr:
                print(f"[Docker Stderr]:\n{result.stderr}")

            if result.returncode == 0 and result.stdout:
                match = re.search(r"===DATA_START===(.*?)===DATA_END===", result.stdout, re.DOTALL)

                if match:
                    b64_data = match.group(1).strip()
                    raw_bytes = base64.b64decode(b64_data)

                    with np.load(io.BytesIO(raw_bytes)) as archive:
                        ver_face = archive['ver_face']
                        R = archive['rotation']
                        ver_landmarks = archive['ver_landmarks']

                    print(f"[Success] Vertex array shape restored: {ver_face.shape}")
                    print(f"[Success] Rotation matrix restored:\n{R}")

                    return {
                        "success": True,
                        "target_vertices": ver_face,
                        "rotation": R,
                        "ver_landmarks": ver_landmarks
                    }
                else:
                    print("[Error] Data markers not found in container stdout.")
                    return {"success": False}
            else:
                return {"success": False}
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
            print(f"[Error] Docker execution failed:\n{err_msg}")
            return {"success": False}
        except Exception as e:
            print(f"[Error] Failed to deserialize data from container: {e}")
            return {"success": False}

    def volumetric_skull_geometry(self, image_path: str, output_obj_path: str) -> bool:
        try:
            extracted_data = self.build_bald_dome(image_path, output_obj_path)
            if not extracted_data.get("success", False):
                print("[Error] Could not extract face features for skull generation.")
                return False

            print("[Info] Starting volumetric skull geometry calculation via FLAME fitting...")

            target_vertices = extracted_data.get("target_vertices")
            if target_vertices is not None:
                print(f"[Debug] target_vertices shape: {target_vertices.shape}")
            ver_landmarks = extracted_data.get("ver_landmarks")
            if ver_landmarks is not None:
                print(f"[Debug] ver_landmarks shape: {ver_landmarks.shape}")
            rotation = extracted_data.get("rotation")

            try:
                from generators.flame_fitting import fit_flame_to_vertices
                result = fit_flame_to_vertices(target_vertices=target_vertices,
                    ver_landmarks=ver_landmarks,
                    rotation_matrix=rotation,
                    output_obj_path=output_obj_path,
                    )
                if result is None:
                    print("[Error] FLAME fitting optimization failed.")
                    return False

                print(f"[Success] Volumetric skull geometry successfully saved to: {output_obj_path}")
                return True

            except Exception as e:
                print(f"[Error] Failed during volumetric skull geometry optimization: {e}")
                return False
        except Exception as e:
            print(f"[Error] Failed during volumetric skull geometry optimization: {e}")
            return False

        # finally:
        #     self.clean_up_docker()


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
    builder = FaceGeometryBuilder()

    test_photo = os.path.join(BASE_DIR, "media", "test_face.jpg")
    output_debug_obj = os.path.join(BASE_DIR, "media", "debug", "volumetric_skull.obj")
    os.makedirs(os.path.dirname(output_debug_obj), exist_ok=True)

    builder.volumetric_skull_geometry(test_photo, output_debug_obj)






# python apps/hairstyles/ai_engine/face_geometry.py
