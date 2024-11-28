from .core import PuLIDEncoder, hack_unet, get_unet_attn_layers
from .utils import load_file_weights, state_dict_extract_names
import torch
from .attention_processors import PuLIDAttnProcessor
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
            weights: str | Dict[str, torch.Tensor],
            pulid_encoder: PuLIDEncoder = None,
            use_id_former: bool = True
        ):
            self.unet = hack_unet(self.unet)

            pulid_encoder = PuLIDEncoder(use_id_former=use_id_former) if pulid_encoder is None else pulid_encoder
            pulid_encoder.to(self.device)
            self.pulid_encoder = pulid_encoder

            state_dict = load_file_weights(weights) if isinstance(weights, str) else weights
            state_dict = state_dict_extract_names(state_dict)  
            for module in state_dict:
                if module == "id_adapter" or module == "pulid_encoder":
                    self.pulid_encoder.id_encoder.load_state_dict(state_dict=state_dict[module], strict=False)
                elif module == "id_adapter_attn_layers" or module == "pulid_ca":
                    pulid_attn_layers = get_unet_attn_layers(self.unet)
                    pulid_attn_layers.load_state_dict(state_dict=state_dict[module], strict=False)


        def to(self, device: str):
            super().to(device)
            if hasattr(self, "pulid_encoder"):
                self.pulid_encoder.to(device)
        
        @classmethod
        @wraps(pipeline_constructor.from_pipe)
        def from_pipe(cls, pipeline, **kwargs):
            pipe = super().from_pipe(pipeline, **kwargs)
            if isinstance(pipeline, PuLIDPipeline):
                if hasattr(pipeline, "pulid_encoder"): pipe.pulid_encoder(pipeline.pulid_encoder)
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
                id_embedding = self.pulid_encoder(id_image)
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
                    isinstance(attn_processor, PuLIDAttnProcessor) and isinstance(attn_processor.original_attn_processor, (
                        IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)
                    )
                ): 
                    attn_processor = attn_processor.original_attn_processor if isinstance(
                        attn_processor, PuLIDAttnProcessor
                    ) and isinstance(
                        attn_processor.original_attn_processor, (
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