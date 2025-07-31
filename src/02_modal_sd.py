import io
import random
from typing import Optional

import modal
import torch

CACHE_DIR = "/hf-cache"

# Image の定義. Dockerfile を参照する方法もあります
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate==0.33.0",
        "diffusers==0.31.0",
        "fastapi[standard]==0.115.4",
        "huggingface-hub[hf_transfer]==0.25.2",
        "sentencepiece==0.2.0",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers~=4.44.0",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # faster downloads
            "HF_HUB_CACHE": CACHE_DIR,
        }
    )
)

with image.imports():
    import diffusers
    import torch
    from PIL.Image import Image as PILImage

app = modal.App("example-text-to-image")
MODEL_ID = "adamo1139/stable-diffusion-3.5-large-turbo-ungated"
MODEL_REVISION_ID = "9ad870ac0b0e5e48ced156bb02f85d324b7275d2"

cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

@app.cls(
    image=image,
    gpu="H100",
    timeout=10 * 60,  # 10 minutes
    volumes={CACHE_DIR: cache_volume},
)
class SD:
    @modal.enter()
    def load_pipeline(self):
        self.pipe = diffusers.StableDiffusion3Pipeline.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION_ID,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

    @modal.method()
    def run(
        self, prompt: str, batch_size: int = 4, seed: Optional[int] = None
    ) -> list[PILImage]:
        seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        print("seeding RNG with", seed)
        torch.manual_seed(seed)
        images = self.pipe(
            prompt,
            num_images_per_prompt=batch_size,  # outputting multiple images per prompt is much cheaper than separate calls
            num_inference_steps=4,  # turbo is tuned to run in four steps
            guidance_scale=0.0,  # turbo doesn't use CFG
            max_sequence_length=512,  # T5-XXL text encoder supports longer sequences, more complex prompts
        ).images  # type: ignore
        torch.cuda.empty_cache()  # reduce fragmentation
        return images # type: ignore


"""command:
uv run modal deploy -m src.02_modal_sd
"""
