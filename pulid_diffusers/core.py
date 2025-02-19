from .encoders import IDEncoder, IDFormer, PerceiverAttentionCA
from . import attention_processors
from .utils import img2tensor, tensor2img, to_gray, load_file_weights, state_dict_extract_names

import torch
import gc
import cv2
import numpy as np
import insightface
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize

from typing import Type
from diffusers import FluxTransformer2DModel, DiffusionPipeline

from diffusers.utils import (
    is_accelerate_available,
    is_torch_version,
    logging,
)
from diffusers.models.modeling_utils import load_model_dict_into_meta
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

import torch.nn.functional as F


logger = logging.get_logger(__name__)

from PIL import Image

from eva_clip import create_model_and_transforms
from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


from typing import Dict, Union, Optional, Any

from types import MethodType


class PuLIDFeaturesExtractor():
    def __init__(self):
        self.device = "cpu"
        # preprocessors
        # face align and parsing
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png'
        )
        self.face_helper.face_parse = None
        self.face_helper.face_parse = init_parsing_model(model_name='bisenet', device=self.device)

        # clip-vit backbone
        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)
        self.clip_vision_model = model.visual
        eva_transform_mean = getattr(self.clip_vision_model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(self.clip_vision_model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            eva_transform_mean = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            eva_transform_std = (eva_transform_std,) * 3
        self.eva_transform_mean = eva_transform_mean
        self.eva_transform_std = eva_transform_std

        # antelopev2
        snapshot_download('DIAMONIK7777/antelopev2', local_dir='models/antelopev2')
        self.app = FaceAnalysis(
            name='antelopev2', root='.', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.handler_ante = insightface.model_zoo.get_model('models/antelopev2/glintr100.onnx')
        self.handler_ante.prepare(ctx_id=0)
        gc.collect()
        torch.cuda.empty_cache()

        # other configs
        self.debug_img_list = []

    
    def __call__(self, image: Image):
        image = np.array(image)
        self.face_helper.clean_all()
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # get antelopev2 embedding
        face_info = self.app.get(image_bgr)
        if len(face_info) > 0:
            face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
                -1
            ]  # only use the maximum face
            id_ante_embedding = face_info['embedding']
            self.debug_img_list.append(
                image[
                    int(face_info['bbox'][1]) : int(face_info['bbox'][3]),
                    int(face_info['bbox'][0]) : int(face_info['bbox'][2]),
                ]
            )
        else:
            id_ante_embedding = None
        # using facexlib to detect and align face
        self.face_helper.read_image(image_bgr)
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) == 0:
            raise RuntimeError('facexlib align face fail')
        align_face = self.face_helper.cropped_faces[0]
        # incase insightface didn't detect face
        if id_ante_embedding is None:
            print('fail to detect face using insightface, extract embedding on align face')
            id_ante_embedding = self.handler_ante.get_feat(align_face)
        id_ante_embedding = torch.from_numpy(id_ante_embedding).to(self.device)
        if id_ante_embedding.ndim == 1:
            id_ante_embedding = id_ante_embedding.unsqueeze(0)
        # parsing
        input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
        input = input.to(self.device)
        parsing_out = self.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(input)
        # only keep the face features
        face_features_image = torch.where(bg, white_image, to_gray(input))
        self.debug_img_list.append(tensor2img(face_features_image, rgb2bgr=False))
        # transform img before sending to eva-clip-vit
        face_features_image = resize(face_features_image, self.clip_vision_model.image_size, InterpolationMode.BICUBIC)
        face_features_image = normalize(face_features_image, self.eva_transform_mean, self.eva_transform_std)
        id_cond_vit, id_vit_hidden = self.clip_vision_model(
            face_features_image, return_all_features=False, return_hidden=True, shuffle=False
        )
        id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
        id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)
        id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)
        
        return id_cond, id_vit_hidden

    def to(self, device:str):
        self.device = device
        self.face_helper.device = device
        self.face_helper.face_parse.to(device)
        self.clip_vision_model.to(device)


class PuLIDEncoder:
    def __init__(self,
        use_id_former: bool = True
    ):
        self.device = "cpu"
        self.id_encoder = IDFormer() if use_id_former else IDEncoder()
        self.features_extractor = PuLIDFeaturesExtractor()

    def to(self, device: str):
        self.device = device
        self.id_encoder.to(device)
        self.features_extractor.to(device)

    def get_id_features(self, image: Image):
        return self.features_extractor(image)

    def get_id_embeds(self, face_info_embeds, clip_embeds):
        id_uncond = torch.zeros_like(face_info_embeds)
        id_vit_hidden_uncond = []
        for layer_idx in range(0, len(clip_embeds)):
            id_vit_hidden_uncond.append(torch.zeros_like(clip_embeds[layer_idx]))
        id_embedding = self.id_encoder(face_info_embeds, clip_embeds)
        uncond_id_embedding = self.id_encoder(id_uncond, id_vit_hidden_uncond)
        # return id_embedding
        return torch.cat((uncond_id_embedding, id_embedding), dim=0)

    def __call__(self, image: Image):
        face_info_embeds, clip_embeds = self.features_extractor(image)
        return self.get_id_embeds()
    
    def load_weights(self, weights: str | Dict[str, torch.Tensor]):
        state_dict = load_file_weights(weights) if isinstance(weights, str) else weights
        state_dict = state_dict_extract_names(state_dict)  
        for module in state_dict:
            if module == "id_adapter" or module == "pulid_encoder":
                self.id_encoder.load_state_dict(state_dict[module], strict=True)




def hack_unet(unet):
    id_adapter_attn_procs = {}
    for name, processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

        if isinstance(processor, (attention_processors.PuLIDAttnProcessor, attention_processors.AttnProcessor)):
            id_adapter_attn_procs[name] = processor
            continue

        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is not None:
            id_adapter_attn_procs[name] = attention_processors.PuLIDAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                original_attn_processor=processor
            ).to(unet.device)
        else:
            id_adapter_attn_procs[name] = attention_processors.AttnProcessor()
    unet.set_attn_processor(id_adapter_attn_procs)
    if hasattr(unet, "_convert_ip_adapter_attn_to_diffusers"):
        unet._convert_ip_adapter_attn_to_diffusers = MethodType(convert_pulid_ip_adapter_attn_to_diffusers, unet)
    return unet



def convert_pulid_ip_adapter_attn_to_diffusers(self, state_dicts, low_cpu_mem_usage=False):
        from diffusers.models.attention_processor import (
            IPAdapterAttnProcessor,
            IPAdapterAttnProcessor2_0
        )

        if low_cpu_mem_usage:
            if is_accelerate_available():
                from accelerate import init_empty_weights

            else:
                low_cpu_mem_usage = False
                logger.warning(
                    "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                    " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                    " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                    " install accelerate\n```\n."
                )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        # set ip-adapter cross-attention processors & load state_dict
        attn_procs = {}
        key_id = 1
        for name, attn_proc in self.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.config.block_out_channels[block_id]

            if cross_attention_dim is None or "motion_modules" in name:
                attn_processor_class = self.attn_processors[name].__class__
                attn_procs[name] = attn_processor_class()
            else:
                attn_processor_class = (
                    attention_processors.IPAdapterAttnProcessor2_0
                    if hasattr(F, "scaled_dot_product_attention")
                    else attention_processors.IPAdapterAttnProcessor
                )
                num_image_text_embeds = []
                for state_dict in state_dicts:
                    if "proj.weight" in state_dict["image_proj"]:
                        # IP-Adapter
                        num_image_text_embeds += [4]
                    elif "proj.3.weight" in state_dict["image_proj"]:
                        # IP-Adapter Full Face
                        num_image_text_embeds += [257]  # 256 CLIP tokens + 1 CLS token
                    elif "perceiver_resampler.proj_in.weight" in state_dict["image_proj"]:
                        # IP-Adapter Face ID Plus
                        num_image_text_embeds += [4]
                    elif "norm.weight" in state_dict["image_proj"]:
                        # IP-Adapter Face ID
                        num_image_text_embeds += [4]
                    else:
                        # IP-Adapter Plus
                        num_image_text_embeds += [state_dict["image_proj"]["latents"].shape[1]]



                ip_adapter_attn_proc = attn_processor_class(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=num_image_text_embeds,
                )

                if isinstance(attn_proc, attention_processors.PuLIDAttnProcessor):
                    attn_proc.original_attn_processor = ip_adapter_attn_proc
                else: attn_proc = ip_adapter_attn_proc

                value_dict = {}
                for i, state_dict in enumerate(state_dicts):
                    value_dict.update({f"to_k_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_k_ip.weight"]})
                    value_dict.update({f"to_v_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_v_ip.weight"]})

                if not low_cpu_mem_usage:
                    attn_proc.load_state_dict(value_dict)
                else:
                    device = next(iter(value_dict.values())).device
                    dtype = next(iter(value_dict.values())).dtype
                    load_model_dict_into_meta(attn_proc, value_dict, device=device, dtype=dtype)

                key_id += 2

                attn_procs[name] = attn_proc

        return attn_procs


def hack_flux_transformer(transformer: FluxTransformer2DModel, double_interval=2, single_interval=4):
    num_ca = 19 // double_interval + 38 // single_interval
    if 19 % double_interval != 0:
        num_ca += 1
    if 38 % single_interval != 0:
        num_ca += 1
    pulid_ca = torch.nn.ModuleList([
        PerceiverAttentionCA().to(transformer.device, transformer.dtype) for _ in range(num_ca)
    ])
    transformer.pulid_ca = pulid_ca
    transformer.pulid_double_interval = double_interval
    transformer.pulid_single_interval = single_interval

    transformer.forward = MethodType(pulid_flux_forward, transformer)

    return transformer


def pulid_flux_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        if joint_attention_kwargs is not None and "id_embedding" in joint_attention_kwargs:
            id_embedding = joint_attention_kwargs.pop("id_embedding")
            id_scale = joint_attention_kwargs.pop("id_scale")

        ca_idx = 0

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

            if index_block % self.pulid_double_interval == 0 and id is not None:
                hidden_states = hidden_states + id_scale * self.pulid_ca[ca_idx](id_embedding, hidden_states)
                ca_idx += 1

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    temb,
                    image_rotary_emb,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )
            
            if index_block % self.pulid_single_interval == 0 and id is not None:
                hidden_states = hidden_states + id_scale * self.pulid_ca[ca_idx](id_embedding, hidden_states)
                ca_idx += 1

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)



class PuLIDPipeline:
    pulid_encoder: PuLIDEncoder = None
    _pulid_timestep_to_start: int = None

    def get_id_embeds(self, image: Image):
        if hasattr(self, "pulid_encoder"):
            return self.pulid_encoder(image)
        else: raise NotImplementedError("PuLID is no loaded")

    def load_pulid(self: Type[DiffusionPipeline], 
        weights: str | Dict[str, torch.Tensor],
        pulid_encoder: PuLIDEncoder = None,
        use_id_former: bool = True
    ):
        self._convert_to_pulid()
        pulid_encoder = PuLIDEncoder(use_id_former=use_id_former) if pulid_encoder is None else pulid_encoder
        pulid_encoder.to(self.device)
        self.pulid_encoder = pulid_encoder
        state_dict = load_file_weights(weights) if isinstance(weights, str) else weights
        state_dict = state_dict_extract_names(state_dict)  
        for module in state_dict:
            if module == "id_adapter" or module == "pulid_encoder":
                self.pulid_encoder.id_encoder.load_state_dict(state_dict=state_dict[module], strict=False)
            elif module == "id_adapter_attn_layers" or module == "pulid_ca":
                pulid_attn_layers = self._get_pulid_layers()
                pulid_attn_layers.load_state_dict(state_dict=state_dict[module], strict=False)

    def to(self, device: str):
        super().to(device)
        if hasattr(self, "pulid_encoder"):
            self.pulid_encoder.to(device)



__all__ = [
    "PuLIDFeaturesExtractor",
    "PuLIDEncoder",
    "IDEncoder",
    "IDFormer",
    "hack_unet",
    "hack_flux_transformer",
]