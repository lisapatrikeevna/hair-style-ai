import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


class HairGeneratorService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        # Lightweight local inpainting model
        self.pipe = None

    def _load_model(self):
        if self.pipe is None:
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                dtype=torch.float16 if self.device != "cpu" else torch.float32,
            ).to(self.device)

    def generate(self, image: Image.Image, mask: Image.Image, prompt: str) -> Image.Image:
        self._load_model()

        original_size = image.size

        resized_image = image.resize((512, 512), Image.Resampling.LANCZOS)
        resized_mask = mask.resize((512, 512), Image.Resampling.NEAREST)

        result = self.pipe(
            prompt=prompt,
            image=resized_image,
            mask_image=resized_mask
        ).images[0]

        return result.resize(original_size, Image.Resampling.LANCZOS)


