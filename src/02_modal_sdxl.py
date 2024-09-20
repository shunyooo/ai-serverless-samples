import io
from typing import cast

import modal
import torch
from fastapi import Response
from modal import gpu
from PIL import Image as PILImage

# Image の定義. Dockerfile を参照する方法もあります
sdxl_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    .pip_install(
        "diffusers==0.26.3",
        "invisible_watermark==0.2.0",
        "transformers~=4.38.2",
        "accelerate==0.27.2",
        "safetensors==0.4.2",
    )
)

app = modal.App("stable-diffusion-xl")
BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

@app.cls(gpu=gpu.A10G(), container_idle_timeout=10, image=sdxl_image)
class Model:
    @modal.build()
    def build(self):
        # デプロイ時に実行されるコード. モデルのダウンロードなど
        from huggingface_hub import snapshot_download
        ignore = [
            "*.bin",
            "*.onnx_data",
            "*/diffusion_pytorch_model.safetensors",
        ]
        snapshot_download(
            BASE_MODEL_ID, ignore_patterns=ignore
        )

    @modal.enter()
    def enter(self):
        # 初回のコンテナ起動時に実行されるコード. モデルの読み込みなど
        from diffusers import (
            StableDiffusionXLPipeline as SDXLPipeline,
        )
        load_options = dict(
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            device_map="auto",
        )

        # Load base model
        self.base: SDXLPipeline = cast(SDXLPipeline, SDXLPipeline.from_pretrained(
            BASE_MODEL_ID, **load_options
        ))

    def _inference(self, prompt:str, negative_prompt:str, n_steps:int) -> PILImage.Image:
        image = self.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            num_images_per_prompt=1,
            output_type="pil",
        ).images[0] # type: ignore
        return image


    @modal.method()
    def inference(self, prompt:str, negative_prompt:str, n_steps:int=24):
        # Modal SDK から呼び出し可能なエンドポイント
        return self._inference(prompt, negative_prompt, n_steps)
    
    @modal.web_endpoint(docs=True)
    def web_inference(
        self, prompt:str, negative_prompt:str, n_steps:int=24
    ):
        # Web API から呼び出し可能なエンドポイント
        pil_image = self._inference(prompt, negative_prompt, n_steps)
        return Response(
            content=_to_bytes(pil_image),
            media_type="image/png",
        )

def _to_bytes(image: PILImage.Image) -> bytes:
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        return output.getvalue()


"""command:
uv run modal deploy src.02_modal_sdxl
uv run modal serve src.02_modal_sdxl
"""

