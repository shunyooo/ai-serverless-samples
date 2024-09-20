import io
from typing import cast

import modal
import torch
from modal import gpu
from PIL import Image as PILImage

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

app = modal.App("stable-diffusion-xl-with-memory-snapshot")
BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

@app.cls(gpu=gpu.A10G(), container_idle_timeout=10, image=sdxl_image, enable_memory_snapshot=True)
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

    @modal.enter(snap=True)
    def enter(self):
        # 初回のコンテナ起動時に実行されるコード. モデルの読み込みなど
        from diffusers import (
            StableDiffusionXLPipeline as SDXLPipeline,
        )
        load_options = dict(
            torch_dtype=torch.float32,  # float32 を使用
            use_safetensors=True,
            device_map=None,  # CPUの場合は "auto" ではなく None を指定
        )
        # Load base model
        self.base: SDXLPipeline = cast(SDXLPipeline, SDXLPipeline.from_pretrained(
            BASE_MODEL_ID, **load_options
        )).to("cpu")
        
    @modal.enter(snap=False)
    def setup(self):
        self.base = self.base.to("cuda").to(torch.float16)

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

    
"""command:
uv run modal deploy src.03_modal_sdxl_memory_snapshot
"""
