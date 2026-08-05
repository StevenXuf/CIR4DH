import torch
import fire
import logging

from pathlib import Path
from transformers import set_seed
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from torchmetrics.retrieval import RetrievalRecall
from torch.utils.data import DataLoader

from loading_data import GLAMQueryDataset, GLAMGalleryDataset
from feature_extractor import FeatureExtractor
from utils import get_config, setup_logging


def get_metrics(txt_embed, img_embed, cfg=None, logger=None, target_index=None):
    device = txt_embed.device
    preds = pairwise_cosine_similarity(txt_embed,img_embed).to(device)
    if target_index is not None:
        tars = torch.tensor(target_index).to(device)
    else:
        tars = torch.ones(txt_embed.size(0), dtype=torch.long).diag().to(device)
    index = torch.arange(txt_embed.size(0)).unsqueeze(1).expand(txt_embed.size(0), img_embed.size(0)).to(device)
    results = {}
    for k in cfg.top_ks:
        compute_recall = RetrievalRecall(top_k=k).to(device)
        recall = compute_recall(preds, tars, indexes=index)
        if logger:
            logger.info(f'|Recall@{k}:\t{recall.item()*100:.2f}%|')
        results[f'Recall@{k}'] = recall.item() * 100
    logger.info(f'|AVG:\t{sum(results.values())/len(results.values()):.2f}%|')

def main(**kwargs):
    cfg = get_config(**kwargs)
    set_seed(cfg.seed)

    cfg.paths.log_path = Path(cfg.paths.log_path) / f"{cfg.split}_{cfg.gen_model}_{cfg.extractor_name}_{'with_metadata' if cfg.use_metadata else 'without_metadata'}_{ 'first_hierarchy' if cfg.use_first_hierarchy else 'all_hierarchy' }_{cfg.baseline}_{cfg.seed}.log"
    setup_logging(cfg.paths.log_path)
    logger = logging.getLogger(__name__)
    logger.info(f'Config: {cfg}')

    query_loader = GLAMQueryDataset(cfg, logger).load_dataloader()
    gallery_dataset = GLAMGalleryDataset(cfg, logger)
    gallery_dataloader = DataLoader(gallery_dataset, 
                                    batch_size=cfg.batch_size, 
                                    shuffle=True, 
                                    num_workers=cfg.num_workers,
                                    collate_fn=lambda batch: {
                                        'pil_image': [item['pil_image'] for item in batch],
                                        'target_names': [item['target_names'] for item in batch],
                                    }
                                    )
    
    feature_extractor=FeatureExtractor(cfg, logger)
    extraction_info = feature_extractor.get_features(query_loader, gallery_dataloader)

    get_metrics(extraction_info['query_embed'], extraction_info['target_embed'], cfg, logger, target_index=extraction_info['target_index'])

"""
python metrics_computation.py --gpu_id 1 --split test_ood --use_metadata True --use_first_hierarchy True --gen_model qwen --batch_size 64 --baseline none
"""

if __name__ == '__main__':
    fire.Fire(main)
