import torch
import fire
import logging
import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path
from transformers import set_seed
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from torchmetrics.retrieval import RetrievalRecall
from torch.utils.data import DataLoader
from textwrap import wrap

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
    setup_logging(str(cfg.paths.log_path))
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

    query_embed = extraction_info['query_embed']
    target_embed = extraction_info['target_embed']
    get_metrics(query_embed, target_embed, cfg, logger, target_index=extraction_info['target_index'])

    ref_names = extraction_info['all_ref_names']
    pil_images = extraction_info['all_ref_pil']
    modifiers = extraction_info['all_modifiers']
    materials = extraction_info['all_materials']
    objects = extraction_info['all_objects']
    ground_truth_names = extraction_info['ground_truth_names']
    generated_descriptions = extraction_info['generated_descriptions']

    logger.info("Plotting retrieval results...")
    similarity = pairwise_cosine_similarity(query_embed, target_embed)
    vals, ids = similarity.topk(5, dim=1)

    sample_size = 3
    idx = np.random.choice(ids.size(0), sample_size, replace=False)

    title_font_dict = {'fontsize': 20, 'fontweight': 'bold'}
    text_font_dict = {'fontsize': 18, 'fontweight': 'bold', 'ha': 'center', 'va': 'center'}

    ncols = 5  # Query Image, Top 5 Retrieved Images
    fig, axes = plt.subplots(sample_size*2, ncols, figsize=(5*ncols, 5 * sample_size * 2))
    for j in range(sample_size):
        query_id = idx[j]
        top_results = ids[query_id]

        query_image = pil_images[query_id]
        modifier = "\n".join(wrap(modifiers[query_id], width=25))
        material = "\n".join(wrap(materials[query_id], width=25))
        obj = "\n".join(wrap(objects[query_id], width=25))
        generated_description = "\n".join(wrap(generated_descriptions[query_id], width=25))

        axes[j*2, 0].imshow(query_image)
        axes[j*2, 0].set_title("Query Image", fontdict=title_font_dict)
        axes[j*2, 0].set_ylabel("Query", fontdict=title_font_dict)

        axes[j*2, 1].text(0.5, 0.5, modifier, fontdict=text_font_dict)
        axes[j*2, 1].set_title("Modifier", fontdict=title_font_dict)

        axes[j*2, 2].text(0.5, 0.5, material, fontdict=text_font_dict)
        axes[j*2, 2].set_title("Material", fontdict=title_font_dict)

        axes[j*2, 3].text(0.5, 0.5, obj, fontdict=text_font_dict)
        axes[j*2, 3].set_title("Object", fontdict=title_font_dict)

        axes[j*2, 4].text(0.5, 0.5, generated_description, fontdict=text_font_dict)
        axes[j*2, 4].set_title("Generated Description", fontdict=title_font_dict)

        for i in range(ncols):
            retrieved_image_name = extraction_info['target_names'][top_results[i]]
            for batch in gallery_dataloader:
                if retrieved_image_name in batch['target_names']:
                    img_idx = batch['target_names'].index(retrieved_image_name)
                    retrieved_image = batch['pil_image'][img_idx]
                    break
            axes[j*2+1, i].imshow(retrieved_image)
            axes[j*2+1, i].set_title(f"Top-{i+1}", fontdict=title_font_dict)

            axes[j*2+1, i].set_xticks([])
            axes[j*2+1, i].set_yticks([])
            axes[j*2, i].set_xticks([])
            axes[j*2, i].set_yticks([])

            for spine in list(axes[j*2+1, i].spines.values())+ list(axes[j*2, i].spines.values()):
                spine.set_visible(False)

            if retrieved_image_name in ground_truth_names[query_id]:
                for spine in axes[j*2+1, i].spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor("green")
                    spine.set_linewidth(4)

            if i == 0:
                axes[j*2+1, i].set_ylabel("Retrieved", fontdict=title_font_dict)

    save_path = Path(cfg.paths.figure_path) / f"retrieval_{cfg.split}_{cfg.gen_model}_{cfg.extractor_name}_{'with_metadata' if cfg.use_metadata else 'without_metadata'}_{'first_hierarchy' if cfg.use_first_hierarchy else 'all_hierarchy'}_{cfg.use_translation}_{cfg.seed}.pdf"
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    logger.info(f"Retrieval results plot saved to {save_path}")

"""
python metrics_computation.py --gpu_id 1 --split test_ood --use_metadata True --use_first_hierarchy True --use_translation True --gen_model qwen --batch_size 64 --baseline gen
"""

if __name__ == '__main__':
    fire.Fire(main)
