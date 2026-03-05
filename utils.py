import base64
import io
import PIL
import torch
import gc
import torchvision.transforms.functional as F
import yaml
import torchvision

import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize, Resize, Lambda, InterpolationMode

def show_tensor_images(images_tensor, num_images=8, file_path="output_image_grid.png"):
    # Make a grid from batch
    img_grid = torchvision.utils.make_grid(images_tensor[:num_images], nrow=4)
    
    # Convert to numpy for plotting
    img_grid = img_grid.permute(1, 2, 0).numpy()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img_grid)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def get_default_config(config_path='./config.yaml'):
    with open(config_path,'r') as f:
        return yaml.safe_load(f)
def get_default_omegaconf(config_path='./config.yaml'):
    return OmegaConf.load(config_path)

class TargetPad():
    """
    If an image aspect ratio is above a target ratio, pad the image to match such target ratio.
    For more details see Baldrati et al. 'Effective conditioned and composed image retrieval combining clip-based features.' Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2022).
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')

def targetpad_transform(mean, std, target_ratio: float=1.25, dim: int=224) -> torch.Tensor:
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(dim),
        lambda img: img.convert("RGB"),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])

def convert_pil_to_tensor(list_of_pils, transform=None):
    if transform is not None:
        return [transform(image) for image in list_of_pils]
    else:
        return [ToTensor()(image) for image in list_of_pils]
    
def resize_crop_normalize(tensor_img, size=224, IMAGE_MEAN=None, IMAGE_STD=None):
    # tensor_img: C×H×W in [0,1]
    img = F.resize(tensor_img, [size, size], interpolation=InterpolationMode.BICUBIC)
    img = F.center_crop(img, size)
    img = F.normalize(img, mean=IMAGE_MEAN, std=IMAGE_STD)
    return img

def convert_pil_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def transform_image(image_size, IMAGENET_MEAN=None, IMAGENET_STD=None):
    img_transform = Compose([
        Resize((image_size, image_size),interpolation=InterpolationMode.BICUBIC),
        CenterCrop(image_size),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) if IMAGENET_MEAN is not None and IMAGENET_STD is not None else Lambda(lambda x: x)
    ])
    return img_transform

def delete_models(*models):
    for model in models:
        del model
    gc.collect()
    torch.cuda.empty_cache()

def get_gpu_memory(device):
    torch.cuda.synchronize(device)
    return torch.cuda.memory_allocated(device)