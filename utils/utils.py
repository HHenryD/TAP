import torch
import json
import torch.nn.functional as F
from clip import clip

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

def make_template_custom(dataset, ref_templates):
    # no specific template
    if dataset in ["StanfordCars", "SUN397", "Caltech101", "ImageNet", "ImageNetSketch", "ImageNetV2", "ImageNetA", "ImageNetR"]:
        templates = ref_templates
    elif dataset == "OxfordPets":
        templates = [temp.replace('.', ', ')+'a type of pet.' for temp in ref_templates]
    elif dataset == "OxfordFlowers":
        templates = [temp.replace('.', ', ')+'a type of flower.' for temp in ref_templates]
    elif dataset == "FGVCAircraft":
        templates = [temp.replace('.', ', ')+'a type of aircraft.' for temp in ref_templates]
    elif dataset == "DescribableTextures":
        templates = [temp.replace('a {}', '{}').replace('{}', '{} texture') for temp in ref_templates]
    elif dataset == "EuroSAT":
        templates = [temp.replace('photo', 'centered satellite photo') if 'photo' in temp else temp[-1]+'in a centered satellite photo.' for temp in ref_templates]
    elif dataset == "Food101":
        templates = [temp.replace('.', ', ')+'a type of food.' for temp in ref_templates]
    elif dataset == "UCF101":
        templates = [temp.replace('{}.', 'a person doing {}.') for temp in ref_templates]
    return templates

IMAGENET_TEMPLATES = [
    "a photo of a {}.",
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
]

IMAGENET_TEMPLATES_SELECT = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]

def wordify(string):
    word = string.replace('_', ' ')
    return word

def load_json(filename):
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as fp:
        return json.load(fp)

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

def entropy(logits):
    p = F.softmax(logits, dim=-1)
    log_p = torch.log(p)
    return -torch.sum(p * log_p, dim=-1)