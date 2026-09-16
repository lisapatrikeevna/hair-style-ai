import os
import sys
from pathlib import Path

# Resolve project root (backend/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from apps.hairstyles.ai_engine.pipeline import HairPipelineService


def run_pipeline_test():
    test_user_dir = BASE_DIR / "media" / "users" / "user_test"
    source_photo = BASE_DIR / "media" / "test_face.jpg"

    if not source_photo.exists():
        print(f"Error: Source image not found at {source_photo}")
        return

    print("=== STARTING PIPELINE TEST ===")

    # 1. Initialize Pipeline
    pipeline = HairPipelineService(user_dir=str(test_user_dir))

    # 2. Step 1: Create Bald Canvas from main photo
    base_canvas_path = pipeline.prepare_base_canvas(main_photo_path=str(source_photo))

    # 3. Step 2: Generate new hairstyle over the bald canvas
    prompt = "short pixie haircut, dark brown hair, highly detailed"
    result_path = pipeline.generate_hairstyle(prompt=prompt)

    print("\n=== PIPELINE TEST FINISHED ===")
    print(f"Check base canvas at: {base_canvas_path}")
    print(f"Check final result at: {result_path}")


if __name__ == "__main__":
    run_pipeline_test()