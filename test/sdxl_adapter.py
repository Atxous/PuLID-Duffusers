from pulid.pulid import PuLIDAdapter
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained("SG161222/RealVisXL_V4.0_Lightning")

pulid_pipe = PuLIDAdapter(pipe)