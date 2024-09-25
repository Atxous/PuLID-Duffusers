from .encoders import IDEncoder
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

import gc
import cv2
import insightface
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from huggingface_hub import hf_hub_download, snapshot_download
from insightface.app import FaceAnalysis
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize
import torch.nn.functional as F

from eva_clip import create_model_and_transforms
from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .utils import img2tensor, tensor2img, to_gray

from diffusers import DiffusionPipeline
from typing import Optional

from . import attention_processor as attention

if hasattr(F, "scaled_dot_product_attention"):
    from .attention_processor import AttnProcessor2_0 as AttnProcessor
    from .attention_processor import IDAttnProcessor2_0 as IDAttnProcessor
else:
    from .attention_processor import AttnProcessor, IDAttnProcessor


class PuLIDFeaturesExtractor():
    def __init__(self, device: str = "cpu"):
        self.device = device
        # preprocessors
        # face align and parsing
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=self.device,
        )
        self.face_helper.face_parse = None
        self.face_helper.face_parse = init_parsing_model(model_name='bisenet', device=self.device)

        # clip-vit backbone
        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)
        model = model.visual
        self.clip_vision_model = model.to(self.device)
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

    
    def __call__(self, image):
        """
        Args:
            image: numpy rgb image, range [0, 255]
        """
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



    

class PuLIDMixin:
    def __init__(self, id_features_extractor: PuLIDFeaturesExtractor=None, device: str = "cpu"):
        self.device = device
        # ID encoders
        self.id_adapter = IDEncoder().to(self.device)
        self.features_extractor = PuLIDFeaturesExtractor(device=self.device) if id_features_extractor == None else id_features_extractor

    def load_pulid_weights(self):
        hf_hub_download('guozinan/PuLID', 'pulid_v1.bin', local_dir='models')
        ckpt_path = 'models/pulid_v1.bin'
        state_dict = torch.load(ckpt_path, map_location='cpu')
        state_dict_dict = {}
        for k, v in state_dict.items():
            module = k.split('.')[0]
            state_dict_dict.setdefault(module, {})
            new_k = k[len(module) + 1 :]
            state_dict_dict[module][new_k] = v
        for module in state_dict_dict:
            print(f'loading from {module}')
            getattr(self, module).load_state_dict(state_dict_dict[module], strict=True)

    def hack_unet_attn_layers(self, unet):
        id_adapter_attn_procs = {}
        for name, _ in unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is not None:
                id_adapter_attn_procs[name] = IDAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                ).to(unet.device)
            else:
                id_adapter_attn_procs[name] = AttnProcessor()
        unet.set_attn_processor(id_adapter_attn_procs)
        self.id_adapter_attn_layers = nn.ModuleList(unet.attn_processors.values())

    
    def get_id_embedding(self, face_info_embeds, clip_embeds):
        id_uncond = torch.zeros_like(face_info_embeds)
        id_vit_hidden_uncond = []
        for layer_idx in range(0, len(clip_embeds)):
            id_vit_hidden_uncond.append(torch.zeros_like(clip_embeds[layer_idx]))
        id_embedding = self.id_adapter(face_info_embeds, clip_embeds)
        uncond_id_embedding = self.id_adapter(id_uncond, id_vit_hidden_uncond)
        # return id_embedding
        return torch.cat((uncond_id_embedding, id_embedding), dim=0)

    def set_pulid_mode(mode: str = "fidelity"):
        if mode == 'fidelity':
            attention.NUM_ZERO = 8
            attention.ORTHO = False
            attention.ORTHO_v2 = True
        elif mode == 'extremely style':
            attention.NUM_ZERO = 16
            attention.ORTHO = True
            attention.ORTHO_v2 = False
        else:
            raise ValueError

    
class PuLIDAdapter(PuLIDMixin):
    def __init__(self, pipe: DiffusionPipeline, id_features_extractor: PuLIDFeaturesExtractor=None, device: str = "cpu"):
        self.pipe = pipe
        super().__init__(id_features_extractor=id_features_extractor, device=device)
        self.hack_unet_attn_layers(pipe.unet)
        self.load_pulid_weights()


    def __call__(self, *args, id_image = None, id_scale: float = 1, pulid_mode:str = 'fidelity', **kwargs):
        pulid_cross_attention = {}
        cross_attention_kwargs = kwargs.pop("cross_attention_kwargs", {})

        self.set_pulid_mode(pulid_mode)

        if id_image is not None or id_image.any():
            id_features, id_clip_embeds = self.features_extractor(id_image)
            id_embedding = self.get_id_embedding(id_features, id_clip_embeds)
            pulid_cross_attention = { 'id_embedding': id_embedding, 'id_scale': id_scale }

        return self.pipe(*args, cross_attention_kwargs={**pulid_cross_attention, **cross_attention_kwargs}, **kwargs ) 