from .utils import hack_unet_attn_layers
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline
)
from typing import Type

def sdxl_pipeline_creator(pipeline_constructor: Type[DiffusionPipeline]) -> Type[DiffusionPipeline]:
    
    class PuLIDPipeline(pipeline_constructor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            super().unet = hack_unet_attn_layers(super().unet)


        def __call__(self, *args, id_embedding = None, id_scale: float = 1, cross_attention_kwargs: dict = {}, **kwargs):
            if not id_embedding == None:
                cross_att = { 'id_embedding': id_embedding, 'id_scale': id_scale }
                
            return super().__call__(*args, cross_attention_kwargs={**cross_att, **cross_attention_kwargs}, **kwargs )
    return PuLIDPipeline


# SDXL Pipelines
StableDiffusionXLPuLIDPipeline = sdxl_pipeline_creator(StableDiffusionXLPipeline)
StableDiffusionXLPuLIDImg2ImgPipeline = sdxl_pipeline_creator(StableDiffusionXLImg2ImgPipeline)
StableDiffusionXLPuLIDInpaintPipeline = sdxl_pipeline_creator(StableDiffusionXLInpaintPipeline)
StableDiffusionXLPuLIDControlNetPipeline = sdxl_pipeline_creator(StableDiffusionXLControlNetPipeline)
StableDiffusionXLPuLIDControlNetImg2ImgPipeline = sdxl_pipeline_creator(StableDiffusionXLControlNetImg2ImgPipeline)
StableDiffusionXLPuLIDControlNetInpaintPipeline = sdxl_pipeline_creator(StableDiffusionXLControlNetInpaintPipeline)

__all__ = [
    "StableDiffusionXLPuLIDPipeline",
    "StableDiffusionXLPuLIDImg2ImgPipeline",
    "StableDiffusionXLPuLIDInpaintPipeline",
    "StableDiffusionXLPuLIDControlNetPipeline",
    "StableDiffusionXLPuLIDControlNetImg2ImgPipeline",
    "StableDiffusionXLPuLIDControlNetInpaintPipeline"
]