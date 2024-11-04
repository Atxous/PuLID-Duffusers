from .core import PuLID, hack_unet_ca_layers
import torch
from .encoders import IDEncoder, IDFormer
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline
)
from typing import Type, Optional, Dict
from functools import wraps

def pipeline_creator(pipeline_constructor: Type[DiffusionPipeline]) -> Type[DiffusionPipeline]:
    
    class PuLIDPipeline(pipeline_constructor):
        @wraps(pipeline_constructor.__init__)
        def __init__(self, *args, pulid: Optional[PuLID], **kwargs):
            super().__init__(*args, **kwargs)
            self.pulid = pulid

        def load_pulid(self, weights: str | Dict[str, torch.Tensor], id_encoder: Optional[IDEncoder | IDFormer] = None, use_id_former: bool = True):
            if self.pulid == None:
                self.pulid = PuLID(id_encoder=id_encoder, use_id_former=use_id_former, ca_layers=hack_unet_ca_layers(self.unet))
            self.pulid.load_weights(weights)

        def to(self, device: str):
            super().to(device)
            self.pulid.to(device)
        
        @classmethod
        def from_pipe(self, pipeline, pulid: Optional[PuLID], **kwargs):
            newpipe = super().from_pipe(pipeline, **kwargs)
            newpipe.pulid = pulid if not pulid == None else pipeline.pulid
            return newpipe

        def __call__(self, *args,
            id_image = None,
            id_scale: float = 1,
            pulid_ortho: str = "off",
            pulid_editability: int = 16,
            pulid_mode:Optional[str],
            **kwargs
        ):
            pulid_cross_attention_kwargs = {}
            cross_attention_kwargs = kwargs.pop("cross_attention_kwargs", {})

            if not pulid_mode == None:
                self.pulid.set_mode(pulid_mode)
            else:
                self.pulid.set_editability(pulid_editability)
                self.pulid.set_ortho(pulid_ortho)

            if not id_image == None:
                id_embedding = self.pulid.get_embeddings(id_image)
                pulid_cross_attention_kwargs = { 'id_embedding': id_embedding, 'id_scale': id_scale }

            return super().__call__(*args, cross_attention_kwargs={**pulid_cross_attention_kwargs, **cross_attention_kwargs}, **kwargs )
        
    return PuLIDPipeline


# SDXL Pipelines
StableDiffusionXLPuLIDPipeline = pipeline_creator(StableDiffusionXLPipeline)
StableDiffusionXLPuLIDImg2ImgPipeline = pipeline_creator(StableDiffusionXLImg2ImgPipeline)
StableDiffusionXLPuLIDInpaintPipeline = pipeline_creator(StableDiffusionXLInpaintPipeline)
StableDiffusionXLPuLIDControlNetPipeline = pipeline_creator(StableDiffusionXLControlNetPipeline)
StableDiffusionXLPuLIDControlNetImg2ImgPipeline = pipeline_creator(StableDiffusionXLControlNetImg2ImgPipeline)
StableDiffusionXLPuLIDControlNetInpaintPipeline = pipeline_creator(StableDiffusionXLControlNetInpaintPipeline)

__all__ = [
    "StableDiffusionXLPuLIDPipeline",
    "StableDiffusionXLPuLIDImg2ImgPipeline",
    "StableDiffusionXLPuLIDInpaintPipeline",
    "StableDiffusionXLPuLIDControlNetPipeline",
    "StableDiffusionXLPuLIDControlNetImg2ImgPipeline",
    "StableDiffusionXLPuLIDControlNetInpaintPipeline"
]