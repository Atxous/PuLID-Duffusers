from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline
)

from .utils import pipeline_creator


# SDXL Pipelines
class StableDiffusionXLPuLIDPipeline(pipeline_creator(StableDiffusionXLPipeline)): pass
class StableDiffusionXLPuLIDImg2ImgPipeline(pipeline_creator(StableDiffusionXLImg2ImgPipeline)): pass
class StableDiffusionXLPuLIDInpaintPipeline(pipeline_creator(StableDiffusionXLInpaintPipeline)): pass
class StableDiffusionXLPuLIDControlNetPipeline(pipeline_creator(StableDiffusionXLControlNetPipeline)): pass
class StableDiffusionXLPuLIDControlNetImg2ImgPipeline(pipeline_creator(StableDiffusionXLControlNetImg2ImgPipeline)): pass
class StableDiffusionXLPuLIDControlNetInpaintPipeline(pipeline_creator(StableDiffusionXLControlNetInpaintPipeline)): pass

__all__ = [
    "StableDiffusionXLPuLIDPipeline",
    "StableDiffusionXLPuLIDImg2ImgPipeline",
    "StableDiffusionXLPuLIDInpaintPipeline",
    "StableDiffusionXLPuLIDControlNetPipeline",
    "StableDiffusionXLPuLIDControlNetImg2ImgPipeline",
    "StableDiffusionXLPuLIDControlNetInpaintPipeline"
]