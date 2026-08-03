import io
import base64
import torch
import logging

from omegaconf import OmegaConf
from pathlib import Path


def setup_logging(log_file: str = "logs/run.log") -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(),
        ],
        force=True,
    )

def convert_pil_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def get_gpu_memory(device):
    torch.cuda.synchronize(device)
    return torch.cuda.memory_allocated(device)

def get_config(**kwargs):
    cfg = OmegaConf.load("config.yaml")
    kw_cfg = OmegaConf.create(kwargs)

    cfg = OmegaConf.merge(cfg, kw_cfg)

    return cfg

if __name__=='__main__':
    cfg = get_config()