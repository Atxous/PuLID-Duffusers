from .encoders import IDEncoder, IDFormer
from . import attention
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

from PIL import Image

from eva_clip import create_model_and_transforms
from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


from typing import Dict, Optional


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


class FaceEncoder:
    def __init__(self, 
        id_encoder: IDEncoder | IDFormer = None,
        features_extractor: PuLIDFeaturesExtractor = None,
        use_id_former: bool = True
    ):
        self.device = "cpu"
        if id_encoder == None:
            id_encoder = IDFormer() if use_id_former else IDEncoder()
        self.id_encoder = id_encoder
        self.features_extractor = features_extractor if features_extractor is not None else PuLIDFeaturesExtractor()

    def to(self, device: str):
        self.device = device
        self.id_encoder.to(device)
        self.features_extractor.to(device)

    def __call__(self, image: Image):
        face_info_embeds, clip_embeds = self.features_extractor(image)
        id_uncond = torch.zeros_like(face_info_embeds)
        id_vit_hidden_uncond = []
        for layer_idx in range(0, len(clip_embeds)):
            id_vit_hidden_uncond.append(torch.zeros_like(clip_embeds[layer_idx]))
        id_embedding = self.id_encoder(face_info_embeds, clip_embeds)
        uncond_id_embedding = self.id_encoder(id_uncond, id_vit_hidden_uncond)
        # return id_embedding
        return torch.cat((uncond_id_embedding, id_embedding), dim=0)
    
    def load_weights(self, weights: str | Dict[str, torch.Tensor]):
        state_dict = load_file_weights(weights) if isinstance(weights, str) else weights
        state_dict = state_dict_extract_names(state_dict)  
        for module in state_dict:
            if module == "id_adapter" or module == "pulid_encoder":
                self.id_encoder.load_state_dict(state_dict[module], strict=True)


class PuLID(FaceEncoder):
    def __init__(self, ca_layers: torch.nn.Module, id_encoder: Optional[IDEncoder | IDFormer] = None, features_extractor: PuLIDFeaturesExtractor = None, use_id_former: bool = True):
        super().__init__(id_encoder, features_extractor=features_extractor, use_id_former=use_id_former)
        self.ca_layers = ca_layers
    

    def load_weights(self, weights: str | Dict[str, torch.Tensor]):
        state_dict = load_file_weights(weights) if isinstance(weights, str) else weights
        state_dict = state_dict_extract_names(state_dict)  
        for module in state_dict:
            if module == "id_adapter" or module == "pulid_encoder":
                self.id_encoder.load_state_dict(state_dict=state_dict[module], strict=True)
            elif module == "id_adapter_attn_layers" or module == "pulid_ca":
                self.ca_layers.load_state_dict(state_dict=state_dict[module], strict=True)
            else:
                getattr(self, module).load_state_dict(state_dict=state_dict[module], strict=True)


    def set_mode(self, mode: str):
        if mode == 'fidelity':
            attention.NUM_ZERO = 8
            attention.ORTHO = False
            attention.ORTHO_v2 = True
        elif mode == 'extremely style':
            attention.NUM_ZERO = 16
            attention.ORTHO = True
            attention.ORTHO_v2 = False
        else:
            raise ValueError("Unsupported pulid mode. Use 'fidelity' or 'extremely style'.")
        
    def set_editability(editability: int):
        attention.NUM_ZERO = editability
    
    def set_ortho(ortho: str):
        if ortho == 'v1':
            attention.ORTHO = True
            attention.ORTHO_v2 = False
        elif ortho == 'v2':
            attention.ORTHO = False
            attention.ORTHO_v2 = True
        elif ortho == 'off':
            attention.ORTHO = False
            attention.ORTHO_v2 = False
        else:
            raise ValueError("Unsupported pulid ortho. Use 'v1', 'v2' or 'off'.")
 
    def to(self, device: str):
        super().to(device)
        self.ca_layers.to(device)


def hack_unet_ca_layers(unet):
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
            id_adapter_attn_procs[name] = attention.PuLIDAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
            ).to(unet.device)
        else:
            id_adapter_attn_procs[name] = attention.AttnProcessor()
    unet.set_attn_processor(id_adapter_attn_procs)
    return torch.nn.ModuleList(unet.attn_processors.values())


def get_unet_ca_layers(unet):
    return torch.nn.ModuleList(unet.attn_processors.values())

__all__ = ["PuLID", "PuLIDFeaturesExtractor", "FaceEncoder", "IDEncoder", "IDFormer", "hack_unet_ca_layers", "get_unet_ca_layers"]