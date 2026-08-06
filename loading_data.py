import fire
import logging
import pandas as pd

from pathlib import Path
from PIL import Image
from transformers import set_seed
from torch.utils.data import Dataset, DataLoader

from utils import  get_config, setup_logging

def process_df(cfg, logger):
    df =pd.read_csv(cfg.paths.cir_file, header=0)
    logger.info(f"Initial partition counts: {df['partition'].value_counts().to_dict()}")

    folders = cfg.image_gallery

    non_existing_images = []
    partitions = []
    invalid_triplets = []

    for i in range(len(df)):
        image_list = [img.split(',') for img in df.iloc[i][['id1','id2']].values.flatten()]
        length = list(map(len, image_list))
        cnts = []
        for j in range(len(image_list)):
            non_existing_cnt = 0
            for image in image_list[j]:
                image = image.strip()
                cnt = 0
                for folder in folders:
                    try:
                        with open(f'{cfg.paths.glam_root}/{folder}/{image}.png', 'rb') as f:
                            pass
                    except FileNotFoundError:
                        cnt += 1
                        continue
                if cnt == len(folders):
                    non_existing_cnt += 1
                    non_existing_images.append(image)
            cnts.append(non_existing_cnt)
        if cnts[0] == length[0] or cnts[1] == length[1]:
            invalid_triplets.append(i)
            partitions.append(df.iloc[i]['partition'])

    logger.info(f"Number of non-existing images: {len(set(non_existing_images))}")
    logger.info(f"Partition counts: {pd.Series(partitions).value_counts().to_dict()}")
    logger.info(f"Number of invalid triplets: {len(invalid_triplets)}")

    df = df.drop(index=invalid_triplets, axis=0).reset_index(drop=True)
    logger.info(f"Shape of processed DataFrame: {df.shape}")

    for column in ['id1', 'id2']:
        df[column] = df[column].fillna("").apply(
            lambda value: [item.strip() for item in value.split(",") if item.strip() not in set(non_existing_images)]
        )
    df.fillna("", inplace=True)
    logger.info(f"Partition counts: {df['partition'].value_counts().to_dict()}")

    if cfg.remove_duplicates:
        keys = pd.DataFrame({
                "id1": df["id1"].map(lambda x: tuple(sorted(x))),
                "id2": df["id2"].map(lambda x: tuple(sorted(x))),
                # "query": df["query"],
            })

        df = df.loc[~keys.duplicated(keep="first")].reset_index(drop=True)
        logger.info(f"Partition counts after removing duplicates: {df['partition'].value_counts().to_dict()}")

    return df[df['partition'] == cfg.split].reset_index(drop=True)

class CustomDataset(Dataset):
    def __init__(self, ref_paths, modifiers, target_paths, materials=None, objects=None):
        self.ref_paths = ref_paths
        self.modifiers = modifiers
        self.target_paths = target_paths
        self.materials = materials
        self.objects = objects

    def __len__(self):
        return len(self.modifiers)

    def __getitem__(self, idx):
        ref_path = self.ref_paths[idx][0]
        tar_paths = self.target_paths[idx]
        ref_img = Image.open(ref_path).convert('RGB').resize((224,224), Image.Resampling.BICUBIC)
        target_img = [Image.open(tar_path).convert('RGB').resize((224,224), Image.Resampling.BICUBIC) for tar_path in tar_paths]
        tar_names = [Path(tar_path).stem for tar_path in tar_paths]
        cap = self.modifiers[idx].strip().replace('for', 'to')
        mat = self.materials[idx].strip() if self.materials is not None else None
        obj = self.objects[idx].strip() if self.objects is not None else None
        sample = {'pil_image': ref_img, 
                  'ref_path': ref_path,
                  'target_pil': target_img, 
                  'target_paths': tar_paths,
                  'modifier': cap, 
                  'target_names': tar_names, 
                  'material': mat, 
                  'object': obj}
        
        return sample

def search_image_path(image_names, gallery_folders, glam_root):
    image_paths = []
    for name in image_names:
        for folder in gallery_folders:
            current_path = Path(glam_root) / folder / f"{name}.png"
            if current_path.exists():
                image_paths.append(current_path)
                break
    return image_paths

class GLAMQueryDataset():
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.df = process_df(self.cfg, self.logger)

    def load_dataset(self):
        self.logger.info(f"Loading GLAM {self.cfg.split} query split with {len(self.df)} samples...")
        ref_paths = self.df["id1"].apply(
                                        lambda ids: search_image_path(ids, self.cfg.image_gallery, self.cfg.paths.glam_root)
                                    ).tolist()
        target_paths = self.df["id2"].apply(
                                        lambda ids: search_image_path(ids, self.cfg.image_gallery, self.cfg.paths.glam_root)
                                                ).tolist()
        modifiers = self.df['query'].tolist()
        if self.cfg.use_metadata:
            if self.cfg.use_first_hierarchy:
                self.logger.info("Using first hierarchy from metadata.")
                materials = [material.split('|')[0] for material in self.df['materials_1'].tolist()]
                objects = [obj.split('|')[0] for obj in self.df['ObjectTypes_1'].tolist()]
            else:
                self.logger.info("Using second hierarchy from metadata.")
                materials = self.df['materials_1'].tolist()
                objects = self.df['ObjectTypes_1'].tolist()
        else:
            materials = [''] * len(self.df)
            objects = [''] * len(self.df)

        return CustomDataset(ref_paths, modifiers, target_paths, materials, objects)

    def load_dataloader(self):
        self.logger.info("Loading GLAM dataloader...")

        dataset = self.load_dataset()
        dataloader = DataLoader(dataset, 
                                batch_size=self.cfg.batch_size, 
                                shuffle=True, 
                                num_workers=self.cfg.num_workers,
                                collate_fn=lambda batch: {
                                    'pil_image': [item['pil_image'] for item in batch],
                                    'target_pil': [item['target_pil'] for item in batch],
                                    'target_names': [item['target_names'] for item in batch],
                                    'modifier': [item['modifier'] for item in batch],
                                    'material': [item['material'] for item in batch],
                                    'object': [item['object'] for item in batch]
                                }
                                )
        return dataloader

class GLAMGalleryDataset(Dataset):
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.image_paths = self.load_gallery_image_paths()

    def load_gallery_image_paths(self):
        self.logger.info("Loading gallery image paths...")
        df = pd.read_csv(self.cfg.paths.db_processed_name, sep=',', header=None)
        df = df[df[1]==self.cfg.split].reset_index(drop=True)
        self.logger.info(df[1].value_counts().to_dict())

        unique_names = list(set(df[df[1] == self.cfg.split][0].to_list()))
        self.logger.info(f"Unique names in {self.cfg.split}: {len(unique_names)}.")

        unique_paths = search_image_path(unique_names, self.cfg.image_gallery, self.cfg.paths.glam_root)
        self.logger.info(f"Unique images in {self.cfg.split}: {len(unique_paths)}.")

        return unique_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)
        return {'pil_image': img, 'target_names': img_path.stem}
    

def main(**kwargs):
    cfg = get_config(**kwargs)
    set_seed(cfg.seed)

    setup_logging(cfg.paths.log_path)
    logger = logging.getLogger(__name__)

    query_loader = GLAMQueryDataset(cfg, logger).load_dataloader()
    gallery_dataset = GLAMGalleryDataset(cfg, logger)
    gallery_dataloader = DataLoader(gallery_dataset, 
                                    batch_size=cfg.batch_size, 
                                    shuffle=True, 
                                    num_workers=cfg.num_workers
                                    )


if __name__ == "__main__":
    fire.Fire(main)