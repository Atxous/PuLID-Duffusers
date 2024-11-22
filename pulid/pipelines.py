from .core import PuLID, hack_unet_ca_layers
import torch
from .encoders import IDEncoder, IDFormer
from .attention import PuLIDAttnProcessor
from diffusers import (
    DiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline
)
from typing import Type, Dict
from functools import wraps

from diffusers.loaders.unet_loader_utils import  _maybe_expand_lora_scales
from diffusers.models.attention_processor import  (
        IPAdapterAttnProcessor,
        IPAdapterAttnProcessor2_0
    )


def pipeline_creator(pipeline_constructor: Type[DiffusionPipeline]) -> Type[DiffusionPipeline]:
    
    class PuLIDPipeline(pipeline_constructor):

        def load_pulid(self, 
            weights_or_pulid: PuLID | str | Dict[str, torch.Tensor],
            id_encoder: IDEncoder | IDFormer = None,
            use_id_former: bool = True
        ):
            is_pulid_instance = isinstance(weights_or_pulid, PuLID)

            if is_pulid_instance:
                self.pulid = PuLID(
                    id_encoder=weights_or_pulid.id_encoder,
                    features_extractor=weights_or_pulid.features_extractor,
                    ca_layers=hack_unet_ca_layers(self.unet)
                )
                self.pulid.ca_layers.load_state_dict(torch.nn.ModuleList(self.unet.attn_processors.values()).state_dict())
            else:
                if not hasattr(self, "pulid"):
                    self.pulid = PuLID(id_encoder=id_encoder, use_id_former=use_id_former, ca_layers=hack_unet_ca_layers(self.unet))
                self.pulid.load_weights(weights_or_pulid)
            
            self.pulid.to(self.device)

        def to(self, device: str):
            super().to(device)
            self.pulid.to(device)
        
        @classmethod
        @wraps(pipeline_constructor.from_pipe)
        def from_pipe(cls, pipeline, **kwargs):
            pipe = super().from_pipe(pipeline, **kwargs)
            if isinstance(pipeline, PuLIDPipeline):
                if hasattr(pipeline, "pulid"): pipe.load_pulid(pipeline.pulid)
            else: pipe = cls(**pipe.components)
            return pipe
        
        @wraps(pipeline_constructor.__call__)
        def __call__(self, *args,
            id_image = None,
            id_scale: float = 1,
            pulid_ortho: str = None,
            pulid_editability: int = 16,
            pulid_mode:str = None,
            **kwargs
        ):
            pulid_cross_attention_kwargs = {}
            cross_attention_kwargs = kwargs.pop("cross_attention_kwargs", {})


            if not id_image == None:
                id_embedding = self.pulid(id_image)
                pulid_cross_attention_kwargs = {
                    'id_embedding': id_embedding,
                    'id_scale': id_scale,
                    'pulid_mode': pulid_mode,
                    'pulid_num_zero': pulid_editability,
                    'pulid_ortho': pulid_ortho
                }

            return super().__call__(*args, cross_attention_kwargs={**pulid_cross_attention_kwargs, **cross_attention_kwargs}, **kwargs )
        
        def set_ip_adapter_scale(self, scale):
            """
            Set IP-Adapter scales per-transformer block. Input `scale` could be a single config or a list of configs for
            granular control over each IP-Adapter behavior. A config can be a float or a dictionary.

            Example:

            ```py
            # To use original IP-Adapter
            scale = 1.0
            pipeline.set_ip_adapter_scale(scale)

            # To use style block only
            scale = {
                "up": {"block_0": [0.0, 1.0, 0.0]},
            }
            pipeline.set_ip_adapter_scale(scale)

            # To use style+layout blocks
            scale = {
                "down": {"block_2": [0.0, 1.0]},
                "up": {"block_0": [0.0, 1.0, 0.0]},
            }
            pipeline.set_ip_adapter_scale(scale)

            # To use style and layout from 2 reference images
            scales = [{"down": {"block_2": [0.0, 1.0]}}, {"up": {"block_0": [0.0, 1.0, 0.0]}}]
            pipeline.set_ip_adapter_scale(scales)
            ```
            """
            unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
            if not isinstance(scale, list):
                scale = [scale]
            scale_configs = _maybe_expand_lora_scales(unet, scale, default_scale=0.0)

            for attn_name, attn_processor in unet.attn_processors.items():
                
                if isinstance(
                    attn_processor, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)
                ) or (
                    isinstance(attn_processor, PuLIDAttnProcessor) and isinstance(attn_processor.original_processor, (
                        IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)
                    )
                ): 
                    attn_processor = attn_processor.original_processor if isinstance(
                        attn_processor, PuLIDAttnProcessor
                    ) and isinstance(
                        attn_processor.original_processor, (
                            IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0
                        )
                    ) else attn_processor
                    if len(scale_configs) != len(attn_processor.scale):
                        raise ValueError(
                            f"Cannot assign {len(scale_configs)} scale_configs to "
                            f"{len(attn_processor.scale)} IP-Adapter."
                        )
                    elif len(scale_configs) == 1:
                        scale_configs = scale_configs * len(attn_processor.scale)
                    for i, scale_config in enumerate(scale_configs):
                        if isinstance(scale_config, dict):
                            for k, s in scale_config.items():
                                if attn_name.startswith(k):
                                    attn_processor.scale[i] = s
                        else:
                            attn_processor.scale[i] = scale_config


        
    return PuLIDPipeline


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