import importlib
import math
import os
import random

import cv2
import numpy as np
import torch
from torchvision.utils import make_grid
from transformers import PretrainedConfig
from typing import Dict
import torch.nn.functional as F

from attention_processors import PuLIDAttnProcessor

from diffusers.models.modeling_utils import load_model_dict_into_meta
from diffusers.utils import (
    is_accelerate_available,
    is_torch_version,
    logging,
)
from diffusers import DiffusionPipeline

from typing import Type, Dict
from functools import wraps
from .core import PuLIDEncoder, hack_unet, get_unet_attn_layers
from .utils import load_file_weights, state_dict_extract_names
from .attention_processors import PuLIDAttnProcessor
from diffusers.loaders.unet_loader_utils import  _maybe_expand_lora_scales
from diffusers.models.attention_processor import  (
        IPAdapterAttnProcessor,
        IPAdapterAttnProcessor2_0
    )

logger = logging.get_logger(__name__)

def seed_everything(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def instantiate_from_config(config):
    if "target" not in config:
        if config == '__is_first_stage__' or config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", {}))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def drop_seq_token(seq, drop_rate=0.5):
    idx = torch.randperm(seq.size(1))
    num_keep_tokens = int(len(idx) * (1 - drop_rate))
    idx = idx[:num_keep_tokens]
    seq = seq[:, idx]
    return seq


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":  # noqa RET505
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def resize_numpy_image_long(image, resize_long_edge=768):
    h, w = image.shape[:2]
    if max(h, w) <= resize_long_edge:
        return image
    k = resize_long_edge / max(h, w)
    h = int(h * k)
    w = int(w * k)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


# from basicsr
def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def to_gray(img):
        x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        x = x.repeat(1, 3, 1, 1)
        return x


from safetensors.torch import load_file

def load_file_weights(path: str):
    # Load the file based on its extension
    if path.endswith('.safetensors'):
        return load_file(path)  # Load using safetensors
    elif path.endswith('.bin'):
        return torch.load(path)  # Load using torch (pickle)
    else:
        raise ValueError("Unsupported file format. Use '.safetensors' or '.bin'.")
    
def state_dict_extract_names(state_dict: Dict[str, torch.Tensor]) -> dict:
    state_dict_dict = {}
    for k, v in state_dict.items():
        module = k.split('.')[0]
        state_dict_dict.setdefault(module, {})
        new_k = k[len(module) + 1 :]
        state_dict_dict[module][new_k] = v
    return state_dict_dict



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
