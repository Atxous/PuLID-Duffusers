from diffusers import (
    FluxPipeline,
    FluxImg2ImgPipeline,
    FluxInpaintPipeline,
    FluxControlNetInpaintPipeline,
    FluxControlNetImg2ImgPipeline,
    FluxControlNetPipeline,
    FluxControlPipeline,
    FluxControlImg2ImgPipeline,
    FluxPriorReduxPipeline,
    FluxFillPipeline,
    FluxControlInpaintPipeline
)

from diffusers import DiffusionPipeline

from typing import Type, get_type_hints
from functools import wraps

from .core import hack_flux_transformer, PuLIDPipeline

def flux_pipeline_creator(pipeline_constructor: Type[DiffusionPipeline]) -> Type[DiffusionPipeline]:
    
    class FluxPuLIDPipeline(pipeline_constructor, PuLIDPipeline):

        def _convert_to_pulid(self):
            self.transformer = hack_flux_transformer(self.transformer)

        def _get_pulid_layers(self):
            return self.transformer.pulid_ca
        
        def _set_pulid_avalible(self, avalible: bool):
            self.transformer.is_pulid_avalible = True
  
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
            #pulid_ortho: str = None,
            #pulid_editability: int = 16,
            #pulid_mode:str = None,
            pulid_timestep_to_start: int = 2,
            **kwargs
        ):
            pulid_joint_attention_kwargs = {}
            joint_attention_kwargs = kwargs.pop("joint_attention_kwargs", {})
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
                pulid_joint_attention_kwargs = {
                    'id_embedding': id_embeds,
                    'id_scale': id_scale,
                    #'pulid_mode': pulid_mode,
                    #'pulid_num_zero': pulid_editability,
                    #'pulid_ortho': pulid_ortho
                }

            return super().__call__(
                *args,
                joint_attention_kwargs={**pulid_joint_attention_kwargs, **joint_attention_kwargs},
                callback_on_step_end=step_callback,
                **kwargs
            )
        
    FluxPuLIDPipeline.__call__.__annotations__ = {**get_type_hints(pipeline_constructor.__call__), **{
        'id_image': None,
        'id_scale': float,
        #'pulid_ortho': str,
        #'pulid_editability': int,
        #'pulid_mode': str,
        'pulid_timestep_to_start': int,
    }}
        
    return FluxPuLIDPipeline



# Flux Pipelines
class FluxPuLIDPipeline(flux_pipeline_creator(FluxPipeline)): pass
class FluxPuLIDImg2ImgPipeline(flux_pipeline_creator(FluxImg2ImgPipeline)): pass
class FluxPuLIDInpaintPipeline(flux_pipeline_creator(FluxInpaintPipeline)): pass
class FluxPuLIDControlNetPipeline(flux_pipeline_creator(FluxControlNetPipeline)): pass
class FluxPuLIDControlNetImg2ImgPipeline(flux_pipeline_creator(FluxControlNetImg2ImgPipeline)): pass
class FluxPuLIDControlNetInpaintPipeline(flux_pipeline_creator(FluxControlNetInpaintPipeline)): pass
class FluxPuLIDControlPipeline(flux_pipeline_creator(FluxControlPipeline)): pass
class FluxPuLIDControlImg2ImgPipeline(flux_pipeline_creator(FluxControlImg2ImgPipeline)): pass
class FluxPuLIDPriorReduxPipeline(flux_pipeline_creator(FluxPriorReduxPipeline)): pass
class FluxPuLIDControlInpaintPipeline(flux_pipeline_creator(FluxControlInpaintPipeline)): pass
class FluxPuLIDFillPipeline(flux_pipeline_creator(FluxFillPipeline)): pass


__all__ = [
    "FluxPuLIDPipeline",
    "FluxPuLIDImg2ImgPipeline",
    "FluxPuLIDInpaintPipeline",
    "FluxPuLIDControlNetPipeline",
    "FluxPuLIDControlNetImg2ImgPipeline",
    "FluxPuLIDControlNetInpaintPipeline",
    "FluxPuLIDControlPipeline",
    "FluxPuLIDControlImg2ImgPipeline",
    "FluxPuLIDPriorReduxPipeline",
    "FluxPuLIDControlInpaintPipeline",
    "FluxPuLIDFillPipeline"
]