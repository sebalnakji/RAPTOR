from pydantic import BaseModel
import io
import logging
import torch
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline

logging.basicConfig(level='DEBUG', format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
logger = logging.getLogger(__name__)
from fastapi import FastAPI, Response
app = FastAPI()

# model_path = "/stable_diffusion/final1_sd_xl_base_1.0.safetensors"
# config 파일
# config_path = "/stable_diffusion/model_index.json"

model_path = "/stable_diffusion/final1_2_sd_xl_base_1.0.safetensors"
config_path = "/stable_diffusion/models_index.json"

# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

# 파이프라인 로드
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    config_file=config_path
)

# GPU 사용 설정 (가능한 경우)
pipe.to("cuda:0")

class stable_diffusion_input(BaseModel):
    user: str = 'a photo of an astronaut riding a horse on mars'

@app.post('/stable_diffusion_inference')
async def image_inference(params: stable_diffusion_input):
    logger.info("이미지 inference 시작")
    logger.info(f"text : {params.user}")

    image = pipe(params.user).images[0] 
    logger.info(type(image))
    imgio = io.BytesIO()
    image.save(imgio, format='PNG')
    logger.info(type(imgio.getvalue()))

    return Response(content=imgio.getvalue(), media_type="image/png")


# uvicorn stable_diffusion_xl:app --host 0.0.0.0 --port 8001

# curl -X POST \
#   http://127.0.0.1:8000/stable_diffusion_inference \
#   -H 'Content-Type: application/json' \
#   -d '{"user": "a photo of an astronaut riding a horse on mars"}'