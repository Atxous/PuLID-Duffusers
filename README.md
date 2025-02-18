# PuLID-Diffusers

**PuLID-Diffusers** is a library that integrates **PuLID** with Hugging Face's **Diffusers**, providing new pipelines adapted for diffusion models like **SDXL** and **FLUX**. This allows identity consistency in generated images while maintaining compatibility with **ControlNet** and **IP-Adapter**.

This project is based on the original **PuLID** repository and follows the **Apache 2.0** license.

## Installation

To install PuLID-Diffusers, use:

```bash
pip install pulid-diffusers
```

## Usage

### FLUX Pipeline

```python
from pulid_diffusers import FluxPuLIDPipeline
from huggingface_hub import hf_hub_download
import torch
from PIL import Image

# Load FLUX model with PuLID support
pipe = FluxPuLIDPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

# Load PuLID checkpoint for FLUX
pipe.load_pulid(hf_hub_download('guozinan/PuLID', 'pulid_flux_v0.9.1.safetensors'))

# Load identity image
face = Image.open("face.jpg")

# Generate images
images = pipe(
    "b&w, instagram photo, portrait photo of 26 y.o blonde woman, perfect detailed eyes, natural skin, hard shadows, film grain",
    id_image=face,
    id_scale=0.8,  # PuLID intensity (default: 1)
    pulid_timestep_to_start=4,  # Step to insert identity features (default: 2)
    num_samples=4,
    width=512,
    height=768,
    num_inference_steps=30,
    seed=2023
).images
```

### SDXL Pipeline

```python
from pulid_diffusers import StableDiffusionXLPuLIDPipeline
from huggingface_hub import hf_hub_download
import torch
from PIL import Image

# Load SDXL model with PuLID support
pipe = StableDiffusionXLPuLIDPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", variant="fp16", torch_dtype=torch.float16)

# Load PuLID checkpoint for SDXL
pipe.load_pulid(hf_hub_download('guozinan/PuLID', 'pulid_v1.1.safetensors'))

# Load identity image
face = Image.open("face.jpg")

# Generate images
images = pipe(
    "b&w, instagram photo, portrait photo of 26 y.o blonde woman, perfect detailed eyes, natural skin, hard shadows, film grain",
    id_image=face,
    id_scale=0.8,  # PuLID intensity (default: 1)
    pulid_timestep_to_start=4,  # Step to insert identity features (default: 2)
    num_samples=4,
    width=512,
    height=768,
    num_inference_steps=30,
    seed=2023
).images
```

### Load PuLID SDXL v1.0

Because version 1.0 of PuLID does not use an IDFormer, it is necessary to disable it when loading the puLID weights.

```python
pipe.load_pulid(hf_hub_download('guozinan/PuLID', 'pulid_v1.0.bin'), use_id_former=False)
```

## Loading from an existing pipeline

```python
pipe = FluxPuLIDPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
iti_pipe = FluxPuLIDPipeline.from_pipe(pipe)  # PuLID is ready
```

## Advanced Usage (SDXL only)

### Using `pulid_mode`

```python
images = pipe(
    "b&w, instagram photo, portrait photo of 26 y.o blonde woman, perfect detailed eyes, natural skin, hard shadows, film grain",
    id_image=face,
    id_scale=1,
    pulid_mode="extremely style",  # Options: 'fidelity', 'extremely style', or None
    num_samples=4,
    width=512,
    height=768,
    num_inference_steps=30,
    seed=2023
).images
```

### Adjusting `pulid_ortho` and `pulid_editability`

```python
images = pipe(
    "b&w, instagram photo, portrait photo of 26 y.o blonde woman, perfect detailed eyes, natural skin, hard shadows, film grain",
    id_image=face,
    id_scale=1,
    pulid_ortho="v1",  # Options: 'v1', 'v2', or None (default: None)
    pulid_editability=20,  # Same as `pulid_num_zero` (default: 16)
    num_samples=4,
    width=512,
    height=768,
    num_inference_steps=30,
    seed=2023
).images
```

## License

This project is licensed under the **Apache 2.0** License, following the original PuLID repository.
