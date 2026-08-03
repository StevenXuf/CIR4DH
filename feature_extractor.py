import torch
import sys
import time
import logging
import numpy as np

from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModel, set_seed
from torchvision import transforms
from torch.utils.data import DataLoader
from open_clip import create_model_and_transforms, create_model_from_pretrained, get_tokenizer

from generation import load_text_gen_model, generate_text
from utils import get_config, get_gpu_memory, setup_logging
from loading_data import GLAMDataset, GLAMGalleryDataset

sys.path.append('/home/fxu/CrossmodalAlignment/BLIP_Official')
from models.blip_itm import blip_itm

class FeatureExtractor():    
    def __init__(self, cfg, logger, **kwargs):
        self.cfg = cfg
        self.logger = logger
        self.device = torch.device(
            f"cuda:{self.cfg.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        self.extractor_name = kwargs.get('extractor_name', self.cfg.extractor_name).lower()
        self.batch_size = kwargs.get('batch_size', self.cfg.batch_size)
        self.dataset_name = kwargs.get('dataset_name', self.cfg.dataset_name).lower()

        self.logger.info(f'Using {self.extractor_name.upper()} on {self.dataset_name.upper()} dataset on {self.device}.')

        # self.text_gen_model, self.text_gen_processor, self.gen_config = load_text_gen_model(self.cfg, self.logger, **kwargs)
        # self.logger.info(f'Using {self.cfg.gen_model.upper()} for generation.')
        
        if self.extractor_name=='clip':
            ckpt_name = 'openai/clip-vit-base-patch32'
            self.extractor=AutoModel.from_pretrained(ckpt_name)
            self.processor=AutoProcessor.from_pretrained(ckpt_name)
        elif self.extractor_name=='openclip':
            ckpt_name = 'ViT-B-32'
            self.extractor, _, self.processor = create_model_and_transforms(ckpt_name, pretrained='laion2b_s34b_b79k')
            self.tokenizer = get_tokenizer(ckpt_name)
        elif self.extractor_name=='blip':
            model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
            self.extractor = blip_itm(pretrained=model_url, image_size=224, vit='base')
        elif self.extractor_name=='siglip' or self.extractor_name=='siglip2':
            ckpt_name = f'google/{self.extractor_name}-base-patch16-224'
            self.extractor = AutoModel.from_pretrained(ckpt_name)
            self.processor = AutoProcessor.from_pretrained(ckpt_name)
        elif self.extractor_name=='coca':
            ckpt_name = 'coca_ViT-B-32'
            self.extractor, _, self.processor = create_model_and_transforms(ckpt_name, pretrained='mscoco_finetuned_laion2b_s13b_b90k')
            self.tokenizer = get_tokenizer(ckpt_name)
        elif self.extractor_name=='openvision':
            sys.path.append('/home/fxu/RobustnessVLM/OpenVision/src/convert_upload')
            from custom_open_clip import get_tokenizer as openvision_get_tokenizer
            ckpt_name = f'UCSC-VLAA/{self.extractor_name}-vit-base-patch16-224'
            self.extractor, self.preprocessor = create_model_from_pretrained(f'hf-hub:{ckpt_name}')
            self.tokenizer = openvision_get_tokenizer(f'hf-hub:{ckpt_name}')
        else:
            raise Exception('No such a extractor available.')
        
        self.logger.info(f'Using {self.extractor_name.upper()} as feature extractor.')
        self.extractor.eval()
        
        init_mem = get_gpu_memory(self.device)
        self.extractor = self.extractor.to(self.device)
        final_mem = get_gpu_memory(self.device)
        self.logger.info(f'Memory usage for loading the model: {(final_mem - init_mem)/1024**2:.2f} MB')

    @torch.no_grad()
    def get_features(self, query_loader, target_loader):

        start_time = time.time()

        self.logger.info(f'Using {self.cfg.gen_model.upper()} for generation.')
        if self.cfg.use_metadata:
            self.logger.warning("Metadata provided and will be used for generation.")
        else:
            self.logger.warning("Metadata not provided or not used. Proceeding without metadata.")

        text_gen_model, text_gen_processor, gen_config = load_text_gen_model(self.cfg, self.logger)

        self.logger.info(f'Extracting features using {self.extractor_name.upper()} on {self.device} for queries...')

        query_embed=[]
        ground_turth_names = []
        for i,batch in enumerate(tqdm(query_loader)):
            pil_images = batch['ref_pil']
            modifiers = batch['modifier']
            materials = batch['material']
            objects = batch['object']
            ground_turth_names.extend(batch['target_names'])
            captions = generate_text(self.cfg, self.logger, pil_images, modifiers, text_gen_model, text_gen_processor, gen_config, materials=materials, objects=objects)
            
            if self.extractor_name == 'clip':
                caption_inputs=self.processor(text=captions,return_tensors='pt',padding=True,truncation=True).to(self.device)
                caption_features=self.extractor.get_text_features(**caption_inputs)
            elif self.extractor_name == 'blip':
                caption_features = []
                for caption in captions:
                    image=torch.randn(1, 3, 224, 224).to(self.device)
                    img_feature,text_feature = self.extractor(image, caption, match_head='itc')
                    caption_features.append(text_feature)
                caption_features = torch.cat(caption_features,dim=0)
            elif self.extractor_name == 'openclip' or self.extractor_name == 'coca':
                text = self.tokenizer(captions).to(self.device)
                caption_features = self.extractor.encode_text(text)
            elif self.extractor_name == 'siglip' or self.extractor_name == 'siglip2':
                image = Image.new('RGB', (224, 224), (0, 0, 0))  # Dummy image for processing
                inputs=self.processor(images=image, text=captions, padding="max_length", max_length=64, truncation=True, return_tensors='pt').to(self.device)
                outputs = self.extractor(**inputs)
                caption_features = outputs.text_embeds
            elif self.extractor_name == 'openvision':
                text = self.tokenizer(captions, context_length=self.extractor.context_length).to(self.device)
                caption_features = self.extractor.encode_text(text)
            else:
                raise Exception('No such a extractor available.')
            
            query_embed.append(caption_features)

        target_embed=[]
        gallery_names = []
        self.logger.info(f'Extracting features using {self.extractor_name.upper()} on {self.device} for targets...')

        for i,batch in enumerate(tqdm(target_loader)):
            pil_images = batch['image']
            gallery_names.extend(batch['image_name'])
            
            if self.extractor_name == 'clip':
                img_inputs=self.processor(images=pil_images,return_tensors='pt').to(self.device)
                img_features=self.extractor.get_image_features(**img_inputs)       
            elif self.extractor_name == 'blip':
                img_features = []
                for image in pil_images:
                    image=transforms.ToTensor()(image).unsqueeze(0).to(self.device)
                    img_feature,text_feature = self.extractor(image, '', match_head='itc')
                    img_features.append(img_feature)
                img_features = torch.cat(img_features,dim=0)
            elif self.extractor_name == 'openclip' or self.extractor_name == 'coca':
                image = torch.cat([self.processor(img).unsqueeze(0) for img in pil_images],dim=0).to(self.device)
                img_features = self.extractor.encode_image(image)
            elif self.extractor_name == 'siglip' or self.extractor_name == 'siglip2':
                inputs=self.processor(images=pil_images, text=[''], padding="max_length", max_length=64, truncation=True, return_tensors='pt').to(self.device)
                outputs = self.extractor(**inputs)
                img_features = outputs.image_embeds
            elif self.extractor_name == 'openvision':
                img_inputs = torch.cat([self.preprocessor(img).unsqueeze(0) for img in pil_images],dim=0).to(self.device)
                img_features = self.extractor.encode_image(img_inputs)
            else:
                raise Exception('No such a extractor available.')
            
            target_embed.append(img_features)
            
        target_embed = torch.cat(target_embed)
        query_embed = torch.cat(query_embed)
        end_time = time.time()
        total_time = end_time - start_time
        sample_time = total_time / len(query_loader.dataset)
        self.logger.info(f'Feature extraction completed in {total_time:.2f}s.')
        self.logger.info(f'Average time per sample: {sample_time:.2f}s.')
    
        self.logger.info(f'Image embed size: {target_embed.size()}')
        self.logger.info(f'Query embed size: {query_embed.size()}')

        target_index = torch.zeros(len(query_embed), len(target_embed), dtype=torch.long)
        for idx in range(len(ground_turth_names)):
            ground_name = ground_turth_names[idx]
            target_index[idx] = torch.from_numpy(np.isin(gallery_names, ground_name).astype(np.int64))

        return {
                'query_embed': query_embed, 
                'target_embed': target_embed, 
                'ground_turth_names': ground_turth_names,
                'target_names': gallery_names,
                'target_index': target_index,
                'total_time': total_time, 
                'sample_time': sample_time}

def main(**kwargs):
    cfg = get_config(**kwargs)
    set_seed(cfg.seed)

    setup_logging(cfg.paths.log_path)
    logger = logging.getLogger(__name__)

    query_loader = GLAMDataset(cfg, logger).load_dataloader()
    gallery_dataset = GLAMGalleryDataset(cfg, logger)
    gallery_dataloader = DataLoader(gallery_dataset, 
                                    batch_size=cfg.batch_size, 
                                    shuffle=True, 
                                    num_workers=cfg.num_workers,
                                    collate_fn=lambda batch: {
                                        'image': [item['image'] for item in batch],
                                        'image_name': [item['image_name'] for item in batch]
                                    }
                                    )
    
    feature_extractor=FeatureExtractor(cfg,logger,**kwargs)
    extraction_info = feature_extractor.get_features(query_loader, gallery_dataloader)

if __name__=='__main__':
    main()
