import copy
import os.path as osp
import os

import math
from functools import reduce
from operator import mul
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Dropout
from torch.cuda.amp import GradScaler, autocast
from einops import rearrange

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import MetricMeter, AverageMeter

from clip import clip
from clip.model import LayerNorm
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils.utils import load_json, wordify, load_clip_to_cpu, entropy, CUSTOM_TEMPLATES, IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT, make_template_custom

import time
import numpy as np
import datetime
from collections import OrderedDict
from tqdm import tqdm

_tokenizer = _Tokenizer()

def load_gpt_descriptions(dataset_name, classnames):
    '''
        load gpt descriptions for each descriptor
        {'descriptor1': {class1: [description1, description2, ...], 'class2': [...]}, 'descriptor2': ...}
    '''
    gpt_descriptions = load_json(f"descriptions/{dataset_name}/descriptions_bag.json")
    classnames_lower = [wordify(c.lower()) for c in classnames]
    gpt_descriptions = {k.lower(): v for k, v in gpt_descriptions.items()}
    descriptors = list(gpt_descriptions.values())[0].keys()
    reordered_gpt_descriptions = {descriptor: {c: [f'a photo of a {des}' for des in gpt_descriptions[c][descriptor]] for c in classnames_lower} for descriptor in descriptors}
    # add cls bag
    cls_temps = IMAGENET_TEMPLATES_SELECT
    cls_des = {c: [f'a photo of a {cls_temp.format(c)}' for cls_temp in cls_temps] for c in classnames_lower}
    reordered_gpt_descriptions['cls'] = cls_des
    print(f"\nExample description for class {classnames_lower[0]}: \"{gpt_descriptions[classnames_lower[0]]}\"\n")
    return reordered_gpt_descriptions

class VisualPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model, num_prompts):
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_dim = clip_model.visual.ln_post.weight.shape[0]
        
        self.deep = cfg.MODEL.TAP.DEEP
        self.deep_layers = 12 # transformer base
        self.dtype = clip_model.dtype
        self.init_prompt(clip_model.visual.conv1.kernel_size)
        
        self.in_dim = clip_model.ln_final.weight.shape[0]
        pooling_cfg = cfg.MODEL.TAP.POOLING
        norm_layer = LayerNorm if pooling_cfg.NORM == 'layernorm' else lambda x: x / x.norm(dim=-1, keepdim=True)
        self.poolings = [(AttentionPooling(
            dim=self.in_dim,
            norm_layer=norm_layer
        )) for _ in range(self.num_prompts+1)] # +1 for cls
    
        self.poolings = nn.ModuleList(self.poolings)
        
    def init_prompt(self, patch_size):
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + self.prompt_dim ))  # noqa

        self.prompt_embeddings = nn.Parameter(torch.zeros(
            1, self.num_prompts, self.prompt_dim ).type(self.dtype))
        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        if self.deep:  # noqa
            total_d_layer = self.deep_layers - 1
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                total_d_layer, self.num_prompts, self.prompt_dim).type(self.dtype))
            nn.init.zeros_(self.deep_prompt_embeddings.data)
        else:
            self.deep_prompt_embeddings = None
    
    def forward(self):
        return self.prompt_embeddings, self.deep_prompt_embeddings

class VisualEncoder(nn.Module):
    def __init__(self, cfg, clip_model, num_prompts):
        super().__init__()
        self.input_resolution = clip_model.visual.input_resolution
        self.output_dim = clip_model.visual.output_dim
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre =  clip_model.visual.ln_pre

        self.transformer = clip_model.visual.transformer

        self.ln_post = clip_model.visual.ln_post
        self.proj = clip_model.visual.proj

        self.deep = cfg.MODEL.TAP.DEEP
        self.prompt_dropout = Dropout(cfg.MODEL.TAP.DROPOUT)
        self.num_prompts = num_prompts
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
    
    def forward_deep_prompt(self, embedding_output, deep_prompt_embeddings):
        hidden_states = None
        # embedding_output is LND
        B = embedding_output.shape[1]
        num_layers = len(self.transformer.resblocks)

        for i in range(num_layers):
            if i == 0:
                hidden_states = self.transformer.resblocks[i](embedding_output)
            else:
                if i <= deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(
                        deep_prompt_embeddings[i-1].expand(B, -1, -1))
                    # LND -> NLD
                    hidden_states = hidden_states.permute(1, 0, 2)

                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_prompt_emb+hidden_states[:, 1:(1+self.num_prompts), :],
                        hidden_states[:, (1+self.num_prompts):, :]
                    ), dim=1)
                    
                    # NLD -> LND
                    hidden_states = hidden_states.permute(1, 0, 2)

                hidden_states = self.transformer.resblocks[i](hidden_states)

        return hidden_states
    
    def prepare_token(self, x, prompt_embeddings):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        B = x.shape[0]
        x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(prompt_embeddings.expand(B, -1, -1)),
                x[:, 1:, :]
            ), dim=1)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        return x

    def forward(self, x, prompt_embeddings, deep_prompt_embeddings, return_img_tokens=False):
        x = self.prepare_token(x, prompt_embeddings)
        if self.deep:
            x = self.forward_deep_prompt(x, deep_prompt_embeddings)
        else:
            x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        cls_feature = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            cls_feature = cls_feature @ self.proj
            
        prompts = self.ln_post(x[:, 1:1+self.num_prompts, :])
        if self.proj is not None:
            prompts = prompts @ self.proj
        return cls_feature, prompts

class CrossAttention(nn.Module):
    def __init__(self, dim=512, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.scale = self.dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.init_weights()
    
    def init_weights(self):
        for child_name, child_module in self.named_children():
            if isinstance(child_module, nn.Linear):
                nn.init.trunc_normal_(child_module.weight, std=0.02)
                if child_module.bias is not None:
                    nn.init.zeros_(child_module.bias)
    
    def forward(self, x, attn_mask):
        B, N, C = x.shape
        k = self.k(x[:, 1:, :]) # [B, N-1, C]
        q = self.q(x[:, 0:1, :]) # [B, 1, C]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1) # [B, 1, N-1]
        # attn_mask: [B, N]
        attn = attn + (1 - attn_mask[:, 1:]).unsqueeze(1) * torch.finfo(attn.dtype).min
        attn = attn.softmax(dim=-1)
        x = attn @ x[:, 1:, :] # [B, 1, C]
        return x

class AttentionPooling(nn.Module):
    def __init__(self, dim, norm_layer=LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim) if isinstance(norm_layer, type) else norm_layer
        self.attn = CrossAttention(dim)
    
    def forward(self, img_token, text_embeddings, attn_mask):
        B = img_token.shape[0]
        num_classes = text_embeddings.shape[1]
        x = torch.cat([img_token.repeat(1, num_classes, 1, 1), text_embeddings.repeat(B, 1, 1, 1)],dim=-2)
        x = rearrange(x, 'b n_c n d -> (b n_c) n d')
        attn_mask = attn_mask.repeat(B, 1, 1)
        attn_mask = rearrange(attn_mask, 'b n_c n -> (b n_c) n')
        x = self.attn(self.norm(x), attn_mask)[0]
        x = rearrange(x, '(b n_c) n d -> b n_c n d', b=B)
        return x.squeeze(2), None

def make_description_batch(descriptor_text_prompt: dict, max_len_quantile=1.0):
    des_len = np.array([v.shape[0] for v in descriptor_text_prompt.values()])
    if max_len_quantile < 1.0:
        max_len = int(np.quantile(des_len, max_len_quantile))
    else:
        max_len = int(np.max(des_len))
    num_classes = len(descriptor_text_prompt)
    batched_descriptor_text_prompt = []
    batched_attn_mask = []
    for idx, (cls_name, text_prompt) in enumerate(descriptor_text_prompt.items()):
        curr_len = text_prompt.shape[0]
        if curr_len > max_len:
            curr_len = max_len
        attn_mask = torch.ones(curr_len)
        batched_descriptor_text_prompt.append(torch.cat([text_prompt[:max_len], torch.zeros(max_len-curr_len, text_prompt.shape[1], dtype=torch.long)]))
        attn_mask = torch.cat([attn_mask, torch.zeros(max_len-curr_len, dtype=torch.long)])
        batched_attn_mask.append(attn_mask)

    batched_descriptor_text_prompt = torch.stack(batched_descriptor_text_prompt)
    batched_attn_mask = torch.stack(batched_attn_mask)

    return batched_descriptor_text_prompt, batched_attn_mask

class TextPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        dataset_name = cfg.DATASET.NAME
        n_cls = len(classnames)
        n_ctx = 4
        ctx_init = "a photo of a"
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # use given words to initialize context vectors
        prompts = clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompts).type(dtype)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]

        print(f'Initial context: "{ctx_init}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.text_prompts = load_gpt_descriptions(dataset_name, classnames)
        self.num_prompts = len(self.text_prompts)-1

    def forward(self, token_prefix, token_suffix, init=False):
        if init:
            ctx = self.ctx.clone().detach().cuda()
        else:
            ctx = self.ctx
        B = token_prefix.shape[0]
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(B, -1, -1)

        prompts = torch.cat(
            [
                token_prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                token_suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts

class TextEncoder(nn.Module):
    def __init__(self, cfg, clip_model, classnames, text_prompt_learner, poolings):
        '''
            Run pooling for each class and each descriptor
            text_prompts: 
                {
                'descriptor_1': {'class_1': ['description_1', 'description_2', ...], 
                           'class_2': ['description_1', 'description_2', ...],
                           ...},
                'descriptor_1': ...,
                }
            ouputs:
                text_embeddings: [batch_size, num_classes, num_descriptors, prompt_dim]
        '''
        super().__init__()
        clip_model.cuda()
        self.text_prompt_learner = text_prompt_learner
        text_prompts = text_prompt_learner.text_prompts
        # {'descriptor1': {class1: [description1, description2, ...], 'class2': [...]}, 'descriptor2': ...}
        self.text_prompts = {description_key: {class_key: clip.tokenize(descriptions) for class_key, descriptions in v.items()} for description_key, v in text_prompts.items()}
        self.poolings = poolings
        self.max_len_quantile = cfg.MODEL.TAP.MAX_LEN_QUANTILE

        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.classnames = classnames
        # save attn masks and text features to avoid recomputing
        self.descriptor_batched_attn_masks = {}
        self.descriptor_text_features = {}
        self.descriptor_token_preffix = {}
        self.descriptor_token_suffix = {}
        self.descriptor_text_prompt = {}
        self.n_ctx = 4

        self.preextract_frozen()
        
    def encode_text(self, text, token_embed=None, embed_only=False):
        if token_embed is None:
            token_embed = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        if embed_only:
            return token_embed
        x = token_embed + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x, token_embed
    
    def sample_text_features(self, text_features, frozen_text_features, attn_mask):
        # 1. Identify non-zero indices in each row
        non_zero_indices = attn_mask.nonzero(as_tuple=True)
        # 2. Randomly sample one index per row where the element is not zero
        sampled_indices = torch.zeros(attn_mask.shape[0], dtype=torch.long).to(attn_mask.device)
        # Iterate over each row to sample an index
        for row_idx in range(attn_mask.shape[0]):
            # Filter non-zero indices for the current row
            row_non_zero_indices = non_zero_indices[1][non_zero_indices[0] == row_idx]
            # Randomly sample one of the non-zero indices
            sampled_index = row_non_zero_indices[torch.randint(0, len(row_non_zero_indices), (1,))]
            sampled_indices[row_idx] = sampled_index

        # 3. Use the sampled indices to gather the corresponding features
        sampled_indices = sampled_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, text_features.shape[-1])
        sampled_features = torch.gather(text_features, 1, sampled_indices)
        sampled_frozen_features = torch.gather(frozen_text_features, 1, sampled_indices)
        sampled_features = sampled_features.squeeze(1)
        sampled_frozen_features = sampled_frozen_features.squeeze(1)
        return sampled_features, sampled_frozen_features
    
    @torch.no_grad()
    def preextract_frozen(self):
        for des_i, (description_key, description_prompts) in enumerate(self.text_prompts.items()):
            # get non-learnable prefix and suffix, can be reused
            if description_key not in self.descriptor_batched_attn_masks:
                batched_descriptor_text_prompt, batched_attn_mask = make_description_batch(description_prompts, self.max_len_quantile)
                    # [num_classes, max_len, prompt_dim]
                rearranged_text_prompts = rearrange(batched_descriptor_text_prompt, 'n_c n d -> (n_c n) d')
                text_features, token_embeds = self.encode_text(rearranged_text_prompts.cuda())
                text_features = rearrange(text_features, '(n_c n) d -> n_c n d', n_c=len(self.classnames))
                token_embeds = rearrange(token_embeds, '(n_c n) v d -> n_c n v d', n_c=len(self.classnames))
                token_preffix = token_embeds[:,:,:1,:]
                token_suffix = token_embeds[:,:,1+self.n_ctx:,:]
                self.descriptor_batched_attn_masks[description_key] = batched_attn_mask
                self.descriptor_token_preffix[description_key] = token_preffix
                self.descriptor_token_suffix[description_key] = token_suffix
                self.descriptor_text_features[description_key] = text_features
                self.descriptor_text_prompt[description_key] = batched_descriptor_text_prompt # tokenized text prompt
        self.frozen_text_features = copy.deepcopy(self.descriptor_text_features)

    @torch.no_grad()
    def update_text_features(self):
        for des_i, (description_key, description_prompts) in enumerate(self.text_prompts.items()):
            token_preffix = self.descriptor_token_preffix[description_key]
            token_suffix = self.descriptor_token_suffix[description_key]
            rearranged_text_prompts = self.descriptor_text_prompt[description_key]
            token_preffix = rearrange(token_preffix, 'n_c n v d -> (n_c n) v d')
            token_suffix = rearrange(token_suffix, 'n_c n v d -> (n_c n) v d')
            rearranged_text_prompts = rearrange(rearranged_text_prompts, 'n_c n d -> (n_c n) d')
            prompts = self.text_prompt_learner(token_preffix, token_suffix)
            text_features, _ = self.encode_text(rearranged_text_prompts, prompts)
            text_features = rearrange(text_features, '(n_c n) d -> n_c n d', n_c=len(self.classnames))
            self.descriptor_text_features[description_key] = text_features
    
    def forward(self, prompt_img_features, cls_img_feature, alternating_epoch=0, sampled_classnames=None, old_indices=None):
        '''Note
            We contrast all image features in the batch (B, D) to each of the text feature for each class (C, D), 
            where the text feature can either be the cls text feature or the prompt text feature
            Since we did cross attention / self attention pooling, the text feature in each descriptor is in (B, C, D) since the pooled features are different for different input images
        '''
        device = prompt_img_features.device
        B = prompt_img_features.shape[0]
        if sampled_classnames is not None:
            text_prompts = {description_key: {class_key: v[class_key] for class_key in sampled_classnames} for description_key, v in self.text_prompts.items()}
            n_c = len(sampled_classnames)
        else:
            text_prompts = self.text_prompts
            n_c = len(self.classnames)
        
        if alternating_epoch == 0 and not self.text_prompt_learner.training:
            self.update_text_features()

        text_embeddings = []
        sampled_text_features = []
        sampled_frozen_text_features = []
        for des_i, (description_key, description_prompts) in enumerate(text_prompts.items()):
            if self.text_prompt_learner.training: 
                # 1. get prompted text features
                batched_attn_mask = self.descriptor_batched_attn_masks[description_key]
                token_preffix = self.descriptor_token_preffix[description_key]
                token_suffix = self.descriptor_token_suffix[description_key]
                rearranged_text_prompts = self.descriptor_text_prompt[description_key]
                frozen_text_features = self.frozen_text_features[description_key]
                if old_indices is not None:
                    batched_attn_mask = batched_attn_mask[old_indices]
                    token_preffix = token_preffix[old_indices]
                    token_suffix = token_suffix[old_indices]
                    rearranged_text_prompts = rearranged_text_prompts[old_indices]
                    frozen_text_features = frozen_text_features[old_indices]
                
                token_preffix = rearrange(token_preffix, 'n_c n v d -> (n_c n) v d')
                token_suffix = rearrange(token_suffix, 'n_c n v d -> (n_c n) v d')
                rearranged_text_prompts = rearrange(rearranged_text_prompts, 'n_c n d -> (n_c n) d')
                prompts = self.text_prompt_learner(token_preffix, token_suffix)
                text_features, _ = self.encode_text(rearranged_text_prompts, prompts)
                text_features = rearrange(text_features, '(n_c n) d -> n_c n d', n_c=n_c)
                # random sample for contrastive loss
                sampled_text_feature, sampled_frozen_text_feature = self.sample_text_features(text_features, frozen_text_features, batched_attn_mask.cuda())
                sampled_text_features.append(sampled_text_feature)
                sampled_frozen_text_features.append(sampled_frozen_text_feature)
            else: # take text features from saved
                text_features = self.descriptor_text_features[description_key].detach()
                batched_attn_mask = self.descriptor_batched_attn_masks[description_key]
                
            # 2. pooling
            batched_attn_mask = torch.cat([torch.ones(n_c, 1, dtype=torch.long), batched_attn_mask], dim=1).to(device)
            pooling_layer = self.poolings[des_i]
            if description_key != 'cls':
                prompt_img_feature = prompt_img_features[:, des_i:des_i+1, :] # [batch_size, 1, feature_dim]
                descriptor_text_features = pooling_layer(prompt_img_feature.unsqueeze(1), text_features.unsqueeze(0), batched_attn_mask.unsqueeze(0))
                descriptor_text_features = descriptor_text_features / descriptor_text_features.norm(dim=-1, keepdim=True)
                text_embeddings.append(descriptor_text_features)
            else:
                descriptor_text_features = pooling_layer(cls_img_feature.unsqueeze(1), text_features.unsqueeze(0), batched_attn_mask.unsqueeze(0))
                descriptor_text_features = descriptor_text_features / descriptor_text_features.norm(dim=-1, keepdim=True)
                cls_embeddings = descriptor_text_features

        text_embeddings = torch.stack(text_embeddings, dim=2)
        if self.text_prompt_learner.training:
            sampled_text_features = torch.stack(sampled_text_features, dim=1)
            sampled_frozen_text_features = torch.stack(sampled_frozen_text_features, dim=1)
            sampled_text_features = rearrange(sampled_text_features, 'n_c n d -> (n_c n) d')
            sampled_frozen_text_features = rearrange(sampled_frozen_text_features, 'n_c n d -> (n_c n) d')
            sampled_labels = torch.arange(sampled_frozen_text_features.shape[0], dtype=torch.long).to(device)
        else:
            sampled_text_features = None
            sampled_labels = None

        return cls_embeddings, text_embeddings, sampled_text_features, sampled_frozen_text_features, sampled_labels

class CustomCLIP_TAP(nn.Module):
    def __init__(self, cfg, clip_model, classnames):
        super().__init__()
        self.classnames = classnames
        self.text_prompt_learner = TextPromptLearner(cfg, classnames, clip_model)
        self.vision_prompt_learner = VisualPromptLearner(cfg, clip_model, self.text_prompt_learner.num_prompts)
        self.text_encoder = TextEncoder(cfg, clip_model, classnames, self.text_prompt_learner, self.vision_prompt_learner.poolings)
        self.visual_encoder = VisualEncoder(cfg, clip_model,  self.text_prompt_learner.num_prompts)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.return_attn = self.text_encoder.return_attn
        self.u1 = cfg.MODEL.TAP.U1
        self.u3 = cfg.MODEL.TAP.U3
        self.u2 = cfg.MODEL.TAP.U2
        self.alpha = cfg.MODEL.TAP.ALPHA
        self.attr_norm_weight = 1/(self.text_prompt_learner.num_prompts+1)
        if self.u1 > 0 or self.u2 > 0:
            self.zs_clip = load_clip_to_cpu(cfg)
            self.zs_clip.float().cuda()
            cls_temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            cls_prompts = [cls_temp.format(c.replace("_", " ")) for c in classnames]
            cls_prompts = torch.cat([clip.tokenize(p) for p in cls_prompts])
            with torch.no_grad():
                cls_embeddings = self.zs_clip.encode_text(cls_prompts.cuda())
                cls_embeddings = cls_embeddings / cls_embeddings.norm(dim=-1, keepdim=True)
            self.cls_text_features = cls_embeddings
    
    def forward(self, image, label=None, alternating_epoch=0):
        if self.vision_prompt_learner.training:
            prompt_embeddings, deep_prompt_embeddings = self.vision_prompt_learner()
            cls_img_features, prompt_img_features = self.visual_encoder(image.type(self.dtype), prompt_embeddings, deep_prompt_embeddings)
        else:
            with torch.no_grad():
                prompt_embeddings, deep_prompt_embeddings = self.vision_prompt_learner()
                cls_img_features, prompt_img_features = self.visual_encoder(image.type(self.dtype), prompt_embeddings, deep_prompt_embeddings)
        
        # if the dataset has too many classes, we sample a subset of classes to train the text prompt learner
        # the sampled classes should include the current batch of classes plus randomly sampled classes to a total of 80 classes
        if len(self.classnames) > 80 and self.text_prompt_learner.training:
            batch_classnames = np.array(self.classnames)[label.cpu()]
            unique_batch_classnames = list(set(batch_classnames)) # find unique classnames in the batch
            remaining_classnames = list(set(self.classnames) - set(unique_batch_classnames))
            sampled_classnames = unique_batch_classnames + np.random.choice(remaining_classnames, 80-len(unique_batch_classnames), replace=False).tolist()
            new_label = torch.tensor([sampled_classnames.index(c) for c in batch_classnames], dtype=torch.long).to(cls_img_features.device)
            old_indices = [self.classnames.index(c) for c in sampled_classnames]
            zs_text_features = self.cls_text_features[old_indices]
            sampled_classnames = [wordify(c.lower()) for c in sampled_classnames]
        else:
            sampled_classnames = None
            new_label = label
            old_indices = None
            zs_text_features = self.cls_text_features

        cls_text_features, prompt_text_features, sampled_text_features, sampled_frozen_text_features, sampled_labels, attn_weights = self.text_encoder(prompt_img_features, cls_img_features.unsqueeze(1), alternating_epoch, sampled_classnames, old_indices)
        # prompt_img_features: [batch_size, num_descriptors, feature_dim]
        # prompt_text_features: [batch_size, num_classes, num_descriptors, feature_dim]
        assert prompt_img_features.shape[1] == prompt_text_features.shape[2]

        logit_scale = self.logit_scale.exp()
        cls_img_features = cls_img_features / cls_img_features.norm(dim=-1, keepdim=True)
        cls_text_features = cls_text_features / cls_text_features.norm(dim=-1, keepdim=True)
        cls_logits = logit_scale * torch.bmm(cls_img_features.unsqueeze(1), cls_text_features.transpose(1,2)).squeeze(1)
        
        prompt_img_features = prompt_img_features / prompt_img_features.norm(dim=-1, keepdim=True)
        prompt_img_features = prompt_img_features.transpose(0,1) # [num_descriptors, batch_size, feature_dim]
        prompt_text_features = prompt_text_features / prompt_text_features.norm(dim=-1, keepdim=True)
        prompt_text_features = prompt_text_features.permute(2,0,1,3) # [num_descriptors, batch_size, num_classes, feature_dim]
        prompt_logits = logit_scale * prompt_img_features.unsqueeze(2)@prompt_text_features.transpose(2,3) # [num_descriptors, batch_size, 1, num_classes]
        prompt_logits = prompt_logits.squeeze(2) # [num_descriptors, batch_size, num_classes]
        all_logits = torch.cat([cls_logits.unsqueeze(0), prompt_logits], 0)

        if self.u1 > 0 or self.u2 > 0:
            with torch.no_grad():
                zs_img_features = self.zs_clip.visual(image.type(self.dtype))
                zs_img_features = zs_img_features / zs_img_features.norm(dim=-1, keepdim=True)
            zs_logits = logit_scale * zs_img_features @ zs_text_features.t()
        else:
            zs_img_features = None
            zs_logits = None
        
        if sampled_text_features is not None:
            assert self.u3 > 0
            sampled_text_features = sampled_text_features / sampled_text_features.norm(dim=-1, keepdim=True)
            sampled_frozen_text_features = sampled_frozen_text_features / sampled_frozen_text_features.norm(dim=-1, keepdim=True)
            tt_logit = logit_scale * sampled_text_features @ sampled_frozen_text_features.t()
            tt_logit_inverse = logit_scale * sampled_frozen_text_features @ sampled_text_features.t()
        else:
            tt_logit = None
            tt_logit_inverse = None

        if self.return_attn:
            return all_logits, attn_weights
        if self.text_prompt_learner.training or self.vision_prompt_learner.training:
            return all_logits, zs_logits, zs_img_features, cls_img_features, tt_logit, tt_logit_inverse, new_label, sampled_labels
        else:
            return self.process_logits(all_logits)

    def process_logits(self, all_logits):
        attr_weights = 0.6/(all_logits.shape[0]-1)
        weights = torch.tensor([self.alpha]+[attr_weights]*(all_logits.shape[0]-1)).reshape(-1,1).type(all_logits.dtype).to(all_logits.device)
        weights = weights.repeat(1, all_logits.shape[1])
        final_logit = torch.einsum('ijk,ij->jk', all_logits, weights.detach())
        return final_logit

@TRAINER_REGISTRY.register()
class TAP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.TAP.PREC == "fp32"
    
    # TODO: reinit optimizer and scheduler: we have one for text and one for vision
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # 1. Build model
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.TAP.PREC == "fp32" or cfg.TRAINER.TAP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP for TAP")
        self.model = CustomCLIP_TAP(cfg, clip_model, classnames)
        self.model.to(self.device)

        # 2. Build optimizer and scheduler
        assert cfg.VISION_OPTIM.MAX_EPOCH + cfg.TEXT_OPTIM.MAX_EPOCH == cfg.OPTIM.MAX_EPOCH
        self.vision_max_epoch = cfg.VISION_OPTIM.MAX_EPOCH
        self.text_max_epoch = cfg.TEXT_OPTIM.MAX_EPOCH
        vision_param_groups = [{'params': self.model.vision_prompt_learner.parameters()}]
        text_param_groups = [{'params': self.model.text_prompt_learner.parameters()}]
        self.vision_optim = build_optimizer(self.model.vision_prompt_learner, cfg.VISION_OPTIM)
        self.text_optim = build_optimizer(self.model.text_prompt_learner, cfg.TEXT_OPTIM)
        self.vision_sched = build_lr_scheduler(self.vision_optim, cfg.VISION_OPTIM)
        self.text_sched = build_lr_scheduler(self.text_optim, cfg.TEXT_OPTIM)
        self.register_model("vision_prompt_learner", self.model.vision_prompt_learner, self.vision_optim, self.vision_sched)
        self.register_model("text_prompt_learner", self.model.text_prompt_learner, self.text_optim, self.text_sched)
        self.scaler = GradScaler() if cfg.TRAINER.TAP.PREC == "amp" else None

        # set up alternating epochs
        self.v2t_alternating_epochs = cfg.MODEL.TAP.V2T_ALTERNATING_EPOCHS
        self.t2v_alternating_epochs = cfg.MODEL.TAP.T2V_ALTERNATING_EPOCHS
        self.alternating_epoch = 0
        self.vision_epoch = 0
        self.text_epoch = 0

        # 3. Start with training vision
        self.current_phase = cfg.MODEL.TAP.STARTING_PHASE
        self.setup_phase()

        # 4. Setup GPA
        self.gpa_mean = cfg.MODEL.TAP.GPA_MEAN
        self.gpa_std = cfg.MODEL.TAP.GPA_STD
        self.setup_gauss()
        self.previous_vision_gpa = None
        self.previous_text_gpa = None
        
    def setup_phase(self):
        print(self.current_phase.replace('_', ' '))
        self.turn_off_grad(phase=self.current_phase)
        if self.current_phase == "train_text":
            self._models["vision_prompt_learner"].eval()
            self._models["text_prompt_learner"].train()
            self.model.u1 = 0.0
            self.model.u3 = self.cfg.MODEL.TAP.U3
        else:
            self._models["vision_prompt_learner"].train()
            self._models["text_prompt_learner"].eval()
            self.model.u1 = self.cfg.MODEL.TAP.U1
            self.model.u3 = 0.0

    def turn_off_grad(self, phase="train_vision"):
        if phase == "train_vision":
            print("Turning off gradients in both the image and the text encoder, as well as the textual prompts")
            for name, param in self.model.named_parameters():
                if "vision_prompt_learner" not in name and "pooling" not in name:
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(True)
                    print(f"Turn on gradients for {name}")
                    param.requires_grad_(True)
        elif phase == "train_text":
            print("Turning off gradients in both the image and the text encoder, as well as the vision prompts")
            for name, param in self.model.named_parameters():
                if "text_prompt_learner" not in name:
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(True)
                    print(f"Turn on gradients for {name}")
                    param.requires_grad_(True)
        else:
            raise NotImplementedError(f"phase {phase} not implemented")

    def train(self):
        """Generic training loops."""
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        test_results = self.after_train()
        return test_results
    
    def after_train(self):
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            test_results = self.test()
        else:
            test_results = None

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()
        return test_results

    def before_epoch(self):
        if self.current_phase == "train_vision":
            if self.alternating_epoch == self.v2t_alternating_epochs:
                self.alternating_epoch = 0
                self.current_phase = "train_text"
                self.setup_phase()
        else:
            if self.alternating_epoch == self.t2v_alternating_epochs:
                self.alternating_epoch = 0
                self.current_phase = "train_vision"
                self.setup_phase()

    def run_epoch(self):
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
        
        self.alternating_epoch += 1
        self.update_gpa()
        if self.current_phase == "train_vision":
            self.vision_epoch += 1
        else:
            self.text_epoch += 1

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        all_logits, zs_logits, zs_img_features, cls_img_features, tt_logit, tt_logit_inverse, new_label, sampled_labels = self.model(image, label, self.alternating_epoch)
        final_logit = self.model.process_logits(all_logits)
        loss_ce = 0.0
        for logit_idx, logit in enumerate(all_logits):
            loss_ce += self.model.attr_norm_weight * F.cross_entropy(logit, new_label)
        loss = loss_ce
        if self.model.u1 > 0:
            loss_scl_image = F.l1_loss(cls_img_features, zs_img_features,
                                    reduction='mean') * self.model.u1
            loss += loss_scl_image
        if self.model.u3 > 0:
            loss_tt = F.cross_entropy(tt_logit, sampled_labels)
            loss_tt_inverse = F.cross_entropy(tt_logit_inverse, sampled_labels)
            loss_tt_total = self.model.u3 * 0.5 * (loss_tt + loss_tt_inverse)
            loss += loss_tt_total
        if self.model.u2 > 0:
            loss_scl_logit = 0.0
            for logit in all_logits:
                loss_scl_logit += self.model.attr_norm_weight * F.kl_div(
                    F.log_softmax(logit / 1, dim=1),
                    F.log_softmax(zs_logits / 1, dim=1),
                    reduction='sum',
                    log_target=True
                ) * (1 * 1) / logit.numel()
            loss += loss_scl_logit * self.model.u2
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss_ce": loss_ce.item()
        }

        if self.model.u1 > 0:
            loss_summary["loss_scl_image"] = loss_scl_image.item()
        
        if self.model.u3 > 0:
            loss_summary["loss_tt"] = loss_tt.item()
            loss_summary["loss_tt_inverse"] = loss_tt_inverse.item()
        
        if self.model.u2 > 0:
            loss_summary["loss_scl_logit"] = loss_scl_logit.item()
        
        loss_summary["acc"] = compute_accuracy(final_logit, new_label)[0].item()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        
        return loss_summary
    
    def get_current_lr(self):
        if self.current_phase == "train_vision":
            name = "vision_prompt_learner"
        else:
            name = "text_prompt_learner"
        return self._optims[name].param_groups[0]["lr"]
    
    def model_backward_and_update(self, loss):
        if self.current_phase == "train_vision":
            self.vision_optim.zero_grad()
            self.detect_anomaly(loss)
            loss.backward()
            self.vision_optim.step()
        else:
            self.text_optim.zero_grad()
            self.detect_anomaly(loss)
            loss.backward()
            self.text_optim.step()

    def update_lr(self):
        if self.current_phase == "train_vision":
            self.vision_sched.step()
        else:
            self.text_sched.step()
    
    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return results
    
    # utils for GPA
    def get_gauss(self, mu, sigma):
        gauss = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return gauss
    
    def setup_gauss(self):
        vision_mean = self.vision_max_epoch * self.gpa_mean
        vision_stdev = self.vision_max_epoch * self.gpa_std
        text_mean = self.text_max_epoch * self.gpa_mean
        text_stdev = self.text_max_epoch * self.gpa_std
        vision_gauss = self.get_gauss(vision_mean, vision_stdev)
        text_gauss = self.get_gauss(text_mean, text_stdev)
        self.vision_gauss = np.array([vision_gauss(a) for a in range(1, self.vision_max_epoch + 1)])
        self.vision_gauss = self.vision_gauss / sum(self.vision_gauss)
        self.text_gauss = np.array([text_gauss(a) for a in range(1, self.text_max_epoch + 1)])
        self.text_gauss = self.text_gauss / sum(self.text_gauss)
    
    def update_gpa(self):
        if self.current_phase == "train_vision":
            curr_vision_epoch_weight = self.vision_gauss[self.vision_epoch]
            curr_vision_model_weight = copy.deepcopy(self.model.vision_prompt_learner.state_dict())
            weighted_vision_state_dict = self.state_dict_weighting(curr_vision_model_weight, curr_vision_epoch_weight)
            if self.previous_vision_gpa is None:
                self.previous_vision_gpa = weighted_vision_state_dict
            else:
                self.previous_vision_gpa = self.state_dict_add(weighted_vision_state_dict, self.previous_vision_gpa)
            if self.vision_epoch == self.vision_max_epoch-1:
                print("Using GPA model for vision final inference...")
                self.model.vision_prompt_learner.load_state_dict(self.previous_vision_gpa)
        else:
            curr_text_epoch_weight = self.text_gauss[self.text_epoch]
            curr_text_model_weight = copy.deepcopy(self.model.text_prompt_learner.state_dict())
            weighted_text_state_dict = self.state_dict_weighting(curr_text_model_weight, curr_text_epoch_weight)
            if self.previous_text_gpa is None:
                self.previous_text_gpa = weighted_text_state_dict
            else:
                self.previous_text_gpa = self.state_dict_add(weighted_text_state_dict, self.previous_text_gpa)
            if self.text_epoch == self.text_max_epoch-1:
                print("Using GPA model for text final inference...")
                self.model.text_prompt_learner.load_state_dict(self.previous_text_gpa)
    
    def state_dict_weighting(self, main_dict, weightage):
        # Average all parameters
        updated_dict = copy.deepcopy(main_dict)
        for key in main_dict:
            updated_dict[key] = main_dict[key] * weightage
        return updated_dict

    def state_dict_add(self, dict1, dict2):
        # Average all parameters
        modified_dict = dict2
        for key in dict1:
            modified_dict[key] = (modified_dict[key] + dict1[key])
        return modified_dict