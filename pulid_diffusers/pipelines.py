from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
)

from diffusers import DiffusionPipeline
from diffusers.loaders.unet_loader_utils import  _maybe_expand_lora_scales
from diffusers.models.attention_processor import  (
        IPAdapterAttnProcessor,
        IPAdapterAttnProcessor2_0
    )

import torch
from typing import Type, get_type_hints
from functools import wraps

from .core import hack_unet, PuLIDPipeline
from .attention_processors import PuLIDAttnProcessor, AttnProcessor

def sd_pipeline_creator(pipeline_constructor: Type[DiffusionPipeline]) -> Type[DiffusionPipeline]:
    
    class StableDiffusionPuLIDPipeline(pipeline_constructor, PuLIDPipeline):

        def _convert_to_pulid(self):
            self.unet = hack_unet(self.unet)

        def _get_pulid_layers(self):
            return torch.nn.ModuleList(self.unet.attn_processors.values())
        
        def _set_pulid_avalible(self, avalible: bool):
            for attn_processor in self._get_pulid_layers():
                if isinstance(attn_processor, PuLIDAttnProcessor):
                    attn_processor.is_pulid_avalible = avalible
  
        @classmethod
        @wraps(pipeline_constructor.from_pipe)
        def from_pipe(cls, pipeline, **kwargs):
            pipe = super().from_pipe(pipeline, **kwargs)
            if isinstance(pipeline, PuLIDPipeline):
                pipe.pulid_encoder = pipeline.pulid_encoder
            else: pipe = cls(**pipe.components)
            return pipe
        
        @wraps(pipeline_constructor.__call__)
        def __call__(self, *args,
            id_image = None,
            id_embeds = None,
            id_scale: float = 1,
            pulid_ortho: str = None,
            pulid_editability: int = 16,
            pulid_mode:str = None,
            pulid_timestep_to_start: int = 2,
            **kwargs
        ):
            pulid_cross_attention_kwargs = {}
            cross_attention_kwargs = kwargs.pop("cross_attention_kwargs", {})
            user_step_callback = kwargs.pop("callback_on_step_end", None)
            step_callback = None

            if not id_image == None or not id_embeds == None: 
                if pulid_timestep_to_start > 0:
                    self._set_pulid_avalible(False)
                    def pulid_step_callback(self, step, timestep, callback_kwargs):
                        if pulid_timestep_to_start >=  step - 1:
                            self._set_pulid_avalible(True)

                        if not user_step_callback == None:
                            return user_step_callback(self, step, timestep, callback_kwargs)
                        else: return callback_kwargs

                    step_callback = pulid_step_callback
                else:
                    self._set_pulid_avalible(True)
                    step_callback = user_step_callback

                if not id_image == None:
                    id_embeds = self.get_id_embeds(id_image).to(self.dtype)
                id_embeds = id_embeds[1:2]
                pulid_cross_attention_kwargs = {
                    'id_embedding': id_embeds,
                    'id_scale': id_scale,
                    'pulid_mode': pulid_mode,
                    'pulid_num_zero': pulid_editability,
                    'pulid_ortho': pulid_ortho
                }

            return super().__call__(
                *args,
                cross_attention_kwargs={**pulid_cross_attention_kwargs, **cross_attention_kwargs},
                callback_on_step_end=step_callback,
                **kwargs
            )
        
        @wraps(pipeline_constructor.set_ip_adapter_scale)
        def set_ip_adapter_scale(self, scale):
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

        @wraps(pipeline_constructor.unload_ip_adapter)
        def unload_ip_adapter(self):
            # remove CLIP image encoder
            if hasattr(self, "image_encoder") and getattr(self, "image_encoder", None) is not None:
                self.image_encoder = None
                self.register_to_config(image_encoder=[None, None])

            # remove feature extractor only when safety_checker is None as safety_checker uses
            # the feature_extractor later
            if not hasattr(self, "safety_checker"):
                if hasattr(self, "feature_extractor") and getattr(self, "feature_extractor", None) is not None:
                    self.feature_extractor = None
                    self.register_to_config(feature_extractor=[None, None])

            # remove hidden encoder
            self.unet.encoder_hid_proj = None
            self.unet.config.encoder_hid_dim_type = None

            # Kolors: restore `encoder_hid_proj` with `text_encoder_hid_proj`
            if hasattr(self.unet, "text_encoder_hid_proj") and self.unet.text_encoder_hid_proj is not None:
                self.unet.encoder_hid_proj = self.unet.text_encoder_hid_proj
                self.unet.text_encoder_hid_proj = None
                self.unet.config.encoder_hid_dim_type = "text_proj"

            # restore original Unet attention processors layers
            attn_procs = {}
            for name, attn_processor in self.unet.attn_processors.items():

                if isinstance(attn_processor, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)):
                    attn_procs[name] = AttnProcessor
                elif isinstance(attn_processor, PuLIDAttnProcessor) and isinstance(attn_processor.original_attn_processor, (
                        IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)
                    ):
                    attn_processor.original_attn_processor = AttnProcessor
                    attn_procs[name] = attn_processor

            self.unet.set_attn_processor(attn_procs)

    StableDiffusionPuLIDPipeline.__call__.__annotations__ = {**get_type_hints(pipeline_constructor.__call__), **{
        'id_image': None,
        'id_scale': float,
        'pulid_ortho': str,
        'pulid_editability': int,
        'pulid_mode': str,
        'pulid_timestep_to_start': int,
    }}
        
    return StableDiffusionPuLIDPipeline


# SDXL Pipelines
class StableDiffusionXLPuLIDPipeline(sd_pipeline_creator(StableDiffusionXLPipeline)): pass
class StableDiffusionXLPuLIDImg2ImgPipeline(sd_pipeline_creator(StableDiffusionXLImg2ImgPipeline)): pass
class StableDiffusionXLPuLIDInpaintPipeline(sd_pipeline_creator(StableDiffusionXLInpaintPipeline)): pass
class StableDiffusionXLPuLIDControlNetPipeline(sd_pipeline_creator(StableDiffusionXLControlNetPipeline)): pass
class StableDiffusionXLPuLIDControlNetImg2ImgPipeline(sd_pipeline_creator(StableDiffusionXLControlNetImg2ImgPipeline)): pass
class StableDiffusionXLPuLIDControlNetInpaintPipeline(sd_pipeline_creator(StableDiffusionXLControlNetInpaintPipeline)): pass


__all__ = [
    "StableDiffusionXLPuLIDPipeline",
    "StableDiffusionXLPuLIDImg2ImgPipeline",
    "StableDiffusionXLPuLIDInpaintPipeline",
    "StableDiffusionXLPuLIDControlNetPipeline",
    "StableDiffusionXLPuLIDControlNetImg2ImgPipeline",
    "StableDiffusionXLPuLIDControlNetInpaintPipeline",
]