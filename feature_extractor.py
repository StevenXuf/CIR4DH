import torch
import sys
import time
import logging
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModel, set_seed
from torchvision import transforms
from torch.utils.data import DataLoader
from open_clip import create_model_and_transforms, create_model_from_pretrained, get_tokenizer

from figure import plot_wordcloud
from generation import load_text_gen_model, generate_text
from utils import get_config, get_gpu_memory, setup_logging
from loading_data import GLAMQueryDataset, GLAMGalleryDataset

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

        if self.extractor_name=='clip':
            ckpt_name = 'openai/clip-vit-base-patch16'
            self.extractor=AutoModel.from_pretrained(ckpt_name)
            self.processor=AutoProcessor.from_pretrained(ckpt_name)
        elif self.extractor_name=='openclip':
            ckpt_name = 'ViT-B-16'
            pretrained='openai'
            self.extractor, _, self.processor = create_model_and_transforms(ckpt_name, pretrained=pretrained)
            self.tokenizer = get_tokenizer(ckpt_name)
        elif self.extractor_name=='blip':
            model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
            self.extractor = blip_itm(pretrained=model_url, image_size=224, vit='base')
        elif self.extractor_name=='siglip' or self.extractor_name=='siglip2':
            ckpt_name = f'google/{self.extractor_name}-base-patch16-224'
            self.extractor = AutoModel.from_pretrained(ckpt_name)
            self.processor = AutoProcessor.from_pretrained(ckpt_name)
        elif self.extractor_name=='coca':
            ckpt_name = 'coca_ViT-B-16'
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
    def get_image_features(self, pil_images):
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
            
        return img_features

    @torch.no_grad()
    def get_text_features(self, captions):
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
        return caption_features

    def get_features(self, query_loader, target_loader):

        start_time = time.time()

        self.logger.info(f'Using {self.cfg.gen_model.upper()} for generation.')
        if self.cfg.use_metadata:
            self.logger.warning("Metadata provided and will be used for generation.")
        else:
            self.logger.warning("Metadata not provided or not used. Proceeding without metadata.")

        if self.cfg.baseline.lower() == 'gen':
            text_gen_model, text_gen_processor, gen_config = load_text_gen_model(self.cfg, self.logger)

        all_materials = []
        all_objects = []

        query_embed=[]
        ground_truth_names = []
        generated_descriptions  = []

        all_modifiers = []
        all_materials = []
        all_objects = []
        all_ref_names = []
        all_ref_pil = []

        self.logger.info(f'Extracting features using {self.extractor_name.upper()} on {self.device} for queries...')
        for batch in tqdm(query_loader):
            pil_images = batch['pil_image']
            modifiers = batch['modifier']
            materials = batch['material']
            objects = batch['object']

            all_modifiers.extend(modifiers)
            all_materials.extend(materials)
            all_objects.extend(objects)
            all_ref_names.extend(batch['ref_name'])
            all_ref_pil.extend(pil_images)

            ground_truth_names.extend(batch['target_names'])
            if self.cfg.baseline == 'text_only':
                query_features = self.get_text_features(modifiers)
            elif self.cfg.baseline == 'image_only':
                query_features = self.get_image_features(pil_images)
            elif self.cfg.baseline == 'combined':
                query_features = F.normalize(self.get_text_features(modifiers) + self.get_image_features(pil_images), p=2, dim=-1)
            elif self.cfg.baseline == 'gen':
                modifiers = generate_text(self.cfg, self.logger, pil_images, modifiers, text_gen_model, text_gen_processor, gen_config, materials=materials, objects=objects)
                query_features = self.get_text_features(modifiers)
                generated_descriptions.extend(modifiers)
            else:
                raise Exception('Choose from "text_only", "image_only", "combined", or "gen".')
            query_embed.append(query_features)

        query_embed = torch.cat(query_embed)

        plot_wordcloud(all_materials, 
                    f"wordcloud_{self.cfg.split}_materials_{self.cfg.seed}",
                    language='ca',
                    custom_stopwords=['material', 'materials'],)
        plot_wordcloud(all_objects, 
                    f"wordcloud_{self.cfg.split}_objects_{self.cfg.seed}",
                    language='ca',
                    )
        
        if self.cfg.baseline != 'gen':
            stopwords = ['change', ',', '.']
            file_name = f"wordcloud_{self.cfg.split}_{self.cfg.baseline}_{self.cfg.seed}"
            plot_wordcloud(all_modifiers, 
                            file_name,
                            custom_stopwords=stopwords)
        else:
            stopwords = ['change', 'changed', 'unchanged', 'generate', 'generated', 'generating', 'feature', 'featuring', ',', '.', 'remain', 'now', 'background', 'foreground', 'object', 'material', 'texture', 'color', 'shape', 'size', 'position', 'orientation', 'angle', 'viewpoint', 'perspective', 'surface', 'pattern', 'style', 'design', 'appearance', 'look', 'feel', 'effect', 'result', 'outcome', 'impact', 'influence', 'consequence', 'ramification', 'implication', 'repercussion', 'made', 'create', 'created', 'creating', 'produce', 'produced', 'producing', 'form', 'formed', 'forming', 'develop', 'developed', 'developing', 'build', 'built', 'building', 'display', 'displayed', 'displaying', 'show', 'showed', 'showing', 'present', 'presented', 'presenting', 'illustrate', 'illustrated', 'illustrating', 'demonstrate', 'demonstrated', 'demonstrating', 'placed', 'place', 'placing', 'positioned', 'position', 'positioning', 'arranged', 'arrange', 'arranging', 'organized', 'organize', 'organizing', 'set', 'setting', 'put', 'putting']
            file_name = f"wordcloud_{self.cfg.split}_{self.cfg.gen_model}_{'with_metadata' if self.cfg.use_metadata else 'without_metadata'}_{ 'first_hierarchy' if self.cfg.use_first_hierarchy else 'all_hierarchy' }_{'en' if self.cfg.use_translation else 'ca'}_{self.cfg.baseline}_{self.cfg.seed}"
            plot_wordcloud(generated_descriptions, 
                            file_name,
                            custom_stopwords=stopwords)

        target_embed=[]
        gallery_names = []
        self.logger.info(f'Extracting features using {self.extractor_name.upper()} on {self.device} for targets...')
        for batch in tqdm(target_loader):
            pil_images = batch['pil_image']
            gallery_names.extend(batch['target_names'])
            target_features = self.get_image_features(pil_images)
            target_embed.append(target_features)
        target_embed = torch.cat(target_embed)
        
        end_time = time.time()
        total_time = end_time - start_time
        sample_time = total_time / len(query_loader.dataset)
        self.logger.info(f'Feature extraction completed in {total_time:.2f}s.')
        self.logger.info(f'Average time per sample: {sample_time:.2f}s.')
    
        self.logger.info(f'Image embed size: {target_embed.size()}')
        self.logger.info(f'Query embed size: {query_embed.size()}')

        target_index = torch.zeros(len(query_embed), len(target_embed), dtype=torch.long)
        for idx in range(len(ground_truth_names)):
            ground_name = ground_truth_names[idx]
            target_index[idx] = torch.from_numpy(np.isin(gallery_names, ground_name).astype(np.int64))

        return {
                'query_embed': query_embed,
                'generated_descriptions': generated_descriptions,
                'target_embed': target_embed, 
                'ground_truth_names': ground_truth_names,
                'target_names': gallery_names,
                'target_index': target_index,
                'total_time': total_time, 
                'sample_time': sample_time,
                'all_modifiers': all_modifiers,
                'all_materials': all_materials,
                'all_objects': all_objects,
                'all_ref_names': all_ref_names,
                'all_ref_pil': all_ref_pil
                }

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
                                    num_workers=cfg.num_workers,
                                    collate_fn=lambda batch: {
                                        'pil_image': [item['pil_image'] for item in batch],
                                        'target_names': [item['target_names'] for item in batch]
                                    }
                                    )
    
    feature_extractor=FeatureExtractor(cfg,logger,**kwargs)
    extraction_info = feature_extractor.get_features(query_loader, gallery_dataloader)

if __name__=='__main__':
    main()
