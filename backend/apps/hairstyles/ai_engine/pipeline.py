# import os
# import shutil
# from PIL import Image
#
# from apps.hairstyles.ai_engine.face_parser import FaceParserService
# from apps.hairstyles.ai_engine.hair_generator import HairGeneratorService
#
#
# class HairPipelineService:
#     def __init__(self, user_dir: str):
#         self.user_dir = user_dir
#         self.sources_dir = os.path.join(user_dir, "sources")
#         self.results_dir = os.path.join(user_dir, "results")
#
#         os.makedirs(self.sources_dir, exist_ok=True)
#         os.makedirs(self.results_dir, exist_ok=True)
#
#         self.generator = HairGeneratorService()
#
#     def prepare_base_canvas(self, main_photo_path: str, ponytail_photo_path: str = None) -> str:
#         """
#         Creates a base canvas for future hairstyles.
#         If ponytail photo exists, uses it directly.
#         Otherwise, removes hair from main photo (inpainting bald scalp).
#         """
#         base_canvas_path = os.path.join(self.sources_dir, "base_canvas.jpg")
#
#         # Scenario 1: Ponytail photo provided -> Use as clean base canvas
#         if ponytail_photo_path and os.path.exists(ponytail_photo_path):
#             shutil.copy(ponytail_photo_path, base_canvas_path)
#             return base_canvas_path
#
#         # Scenario 2: Only main photo provided -> Generate bald head canvas
#         mask = FaceParserService.create_hair_mask(main_photo_path)
#         raw_image = Image.open(main_photo_path).convert("RGB")
#
#         bald_prompt = "bald head, smooth clean scalp, realistic skin texture, neutral background"
#         clean_canvas = self.generator.generate(
#             image=raw_image,
#             mask=mask,
#             prompt=bald_prompt
#         )
#
#         clean_canvas.save(base_canvas_path)
#         return base_canvas_path
#
#     def generate_hairstyle(self, prompt: str) -> str:
#         """
#         Generates new hairstyle over the prepared base canvas.
#         """
#         base_canvas_path = os.path.join(self.sources_dir, "base_canvas.jpg")
#         if not os.path.exists(base_canvas_path):
#             raise FileNotFoundError("Base canvas not found. Run prepare_base_canvas first.")
#
#         base_image = Image.open(base_canvas_path).convert("RGB")
#         mask = FaceParserService.create_hair_mask(base_canvas_path)
#
#         result_image = self.generator.generate(
#             image=base_image,
#             mask=mask,
#             prompt=prompt
#         )
#
#         output_path = os.path.join(self.results_dir, "generated_hairstyle.jpg")
#         result_image.save(output_path)
#         return output_path

import os
import shutil
from PIL import Image

from apps.hairstyles.ai_engine.face_parser import FaceParserService
from apps.hairstyles.ai_engine.hair_generator import HairGeneratorService


class HairPipelineService:
    def __init__(self, user_dir: str):
        print(f"\n[Pipeline] Initializing pipeline for directory: {user_dir}")
        self.user_dir = user_dir
        self.sources_dir = os.path.join(user_dir, "sources")
        self.results_dir = os.path.join(user_dir, "results")

        os.makedirs(self.sources_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        print("[Pipeline] Directories 'sources' and 'results' are ready.")

        print("[Pipeline] Loading Stable Diffusion Generator...")
        self.generator = HairGeneratorService()
        print("[Pipeline] Generator initialized successfully.")

    def prepare_base_canvas(self, main_photo_path: str, ponytail_photo_path: str = None) -> str:
        """
        Creates a base canvas for future hairstyles.
        If ponytail photo exists, uses it directly.
        Otherwise, removes hair from main photo (inpainting bald scalp).
        """
        print("\n==================================================")
        print("[Pipeline: Step 1] Starting Base Canvas Preparation...")
        print("==================================================")

        base_canvas_path = os.path.join(self.sources_dir, "base_canvas.jpg")

        # Scenario 1: Ponytail photo provided
        if ponytail_photo_path and os.path.exists(ponytail_photo_path):
            print(f"[Pipeline: Step 1] Ponytail photo detected at: {ponytail_photo_path}")
            print("[Pipeline: Step 1] Copying ponytail photo directly to base_canvas.jpg...")
            shutil.copy(ponytail_photo_path, base_canvas_path)
            print(f"[Pipeline: Step 1] Base canvas ready: {base_canvas_path}")
            return base_canvas_path

        # Scenario 2: Only main photo provided
        print(f"[Pipeline: Step 1] No ponytail photo found. Processing main photo: {main_photo_path}")

        print("[Pipeline: Step 1.1] Extracting hair mask via MediaPipe...")
        mask = FaceParserService.create_hair_mask(main_photo_path)

        # Save temporary mask for debugging
        debug_mask_path = os.path.join(self.sources_dir, "debug_bald_mask.png")
        mask.save(debug_mask_path)
        print(f"[Pipeline: Step 1.1] Hair mask generated and saved to: {debug_mask_path}")

        print("[Pipeline: Step 1.2] Loading raw source image...")
        raw_image = Image.open(main_photo_path).convert("RGB")

        bald_prompt = "bald head, smooth clean scalp, realistic skin texture, neutral background"
        print(f"[Pipeline: Step 1.3] Running Inpainting to remove hair (Prompt: '{bald_prompt}')...")

        clean_canvas = self.generator.generate(
            image=raw_image,
            mask=mask,
            prompt=bald_prompt
        )

        clean_canvas.save(base_canvas_path)
        print(f"[Pipeline: Step 1] Success! Base bald canvas created at: {base_canvas_path}")
        return base_canvas_path

    def generate_hairstyle(self, prompt: str) -> str:
        """
        Generates new hairstyle over the prepared base canvas.
        """
        print("\n==================================================")
        print(f"[Pipeline: Step 2] Starting Hairstyle Generation...")
        print(f"[Pipeline: Step 2] Requested Prompt: '{prompt}'")
        print("==================================================")

        base_canvas_path = os.path.join(self.sources_dir, "base_canvas.jpg")
        if not os.path.exists(base_canvas_path):
            error_msg = f"Base canvas not found at {base_canvas_path}. Run prepare_base_canvas first."
            print(f"[Pipeline: ERROR] {error_msg}")
            raise FileNotFoundError(error_msg)

        print(f"[Pipeline: Step 2.1] Loading base canvas from: {base_canvas_path}")
        base_image = Image.open(base_canvas_path).convert("RGB")

        print("[Pipeline: Step 2.2] Generating segmentation mask for base canvas...")
        mask = FaceParserService.create_hair_mask(base_canvas_path)

        print("[Pipeline: Step 2.3] Running Stable Diffusion Inpainting for new hairstyle...")
        result_image = self.generator.generate(
            image=base_image,
            mask=mask,
            prompt=prompt
        )

        output_path = os.path.join(self.results_dir, "generated_hairstyle.jpg")
        result_image.save(output_path)
        print(f"[Pipeline: Step 2] Success! Final hairstyle saved to: {output_path}")
        print("==================================================\n")

        return output_path