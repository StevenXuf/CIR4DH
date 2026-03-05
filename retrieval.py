import torch
import fire
import os
import time

from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, LlavaForConditionalGeneration, AutoProcessor, GenerationConfig, set_seed #DeepseekVLForConditionalGeneration,

from feature_extraction import get_feature_extractor, get_metrics
from dataloaders import get_dataloader
from text_to_image_and_text import generate_texts, extract_text_features, extract_image_features
from prompts import get_composed_prompts, get_composed_prompts_other_mllm
from utils import get_default_config, transform_image, delete_models, get_gpu_memory


def get_test_loader(cfg, dataset_name, split, extractor, img_batch_size):
    if dataset_name.lower() == 'circo':
        if split.lower() == 'test':
            test_loader = get_dataloader(cfg, 
                                         mode='classic',
                                         dataset_name=dataset_name, 
                                         batch_size=img_batch_size, 
                                         extractor_name=extractor
                                         )
        elif split.lower() == 'val':
            test_loader = get_dataloader(cfg, 
                                         dataset_name=dataset_name, 
                                         split='val', 
                                         mode='classic',
                                         batch_size=img_batch_size, 
                                         extractor_name=extractor
                                         )
        else:
            raise ValueError(f"Unsupported split: {split} for dataset: {dataset_name}")
    elif dataset_name.lower() == 'fashioniq':
        if split.lower() == 'val' or split.lower() == 'train':
            test_loader = get_dataloader(cfg, 
                                         dataset_name=dataset_name, 
                                         split=split.lower(), 
                                         mode='classic',
                                         batch_size=img_batch_size, 
                                         extractor_name=extractor,
                                         )
        else:
            raise ValueError(f"Unsupported split: {split} for dataset: {dataset_name}")
    elif dataset_name.lower() == 'change_attribute' or dataset_name.lower() == 'focus_attribute' or dataset_name.lower() == 'change_object' or dataset_name.lower() == 'focus_object':
        test_loader = get_dataloader(
            cfg,
            dataset_name=dataset_name,
            extractor_name=extractor,
            batch_size=img_batch_size
        )
    return test_loader

def load_mllm(cfg, **kwargs):
    ### Text Generation Parameters
    
    temperature = kwargs.get('temperature', cfg['TEXT-GENERATION']['GLOBAL']['TEMPERATURE'])
    top_p = kwargs.get('top_p', cfg['TEXT-GENERATION']['GLOBAL']['TOP_P'])
    llm_top_k = kwargs.get('llm_top_k', cfg['TEXT-GENERATION']['GLOBAL']['TOP_K'])
    max_new_tokens = kwargs.get('max_new_tokens', cfg['TEXT-GENERATION']['GLOBAL']['MAX_NEW_TOKENS'])

    device = torch.device(f"cuda") if kwargs.get('device') is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen_config = GenerationConfig(do_sample=True,
                                    temperature=temperature,
                                    top_p=top_p,
                                    top_k=llm_top_k,
                                    max_new_tokens=max_new_tokens
                                )
    
    mllm = kwargs.get('mllm', cfg['GENERAL']['MLLM']).upper()
    before = get_gpu_memory(device)
    if mllm == 'QWEN-7B' or mllm == 'QWEN-3B':
        model_id = cfg['TEXT-GENERATION'][mllm]['MODEL_NAME']
        text_generation_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, 
                                                                                torch_dtype=torch.bfloat16, 
                                                                                device_map={"": device}, 
                                                                                # attn_implementation='flash_attention_2'
                                                                                ).to(device)
    elif mllm == 'LLAVA':
        model_id = cfg['TEXT-GENERATION'][mllm]['MODEL_NAME']
        text_generation_model = LlavaForConditionalGeneration.from_pretrained(model_id, 
                                                                            torch_dtype=torch.bfloat16, 
                                                                            device_map={"": device}
                                                                            ).to(device)
    elif mllm == 'DEEPSEEK':
        model_id = cfg['TEXT-GENERATION']['DEEPSEEK']['MODEL_NAME']
        text_generation_model = DeepseekVLForConditionalGeneration.from_pretrained(model_id, 
                                                                            torch_dtype=torch.bfloat16, 
                                                                            device_map={"": device}
                                                                            ).to(device)
    else:
        raise ValueError(f"Unsupported MLLM: {mllm}. Should be one of ['QWEN', 'LLAVA', 'DEEPSEEK']")
    after = get_gpu_memory(device)
    print(f"Meomeroy taken by {mllm.upper()}: {(after - before) / 1024**2: .2f} MB\n")
    text_generation_model.eval()
    print(f"Using {text_generation_model.__class__.__name__} for text generation with temperature={temperature}, top_p={top_p}, top_k={llm_top_k}, max_new_tokens={max_new_tokens}")
    processor = AutoProcessor.from_pretrained(model_id, 
                                            padding_side='left', 
                                            use_fast=True
                                            )
    return gen_config, processor, text_generation_model

def main(cfg, **kwargs):
    ### General Parameters
    seed = kwargs.get('seed', cfg['GENERAL']['SEED'])
    top_k = kwargs.get('top_k', cfg['GENERAL']['TOP_K'])
    task = kwargs.get('task', cfg['GENERAL']['TASK'])
    device = torch.device(f"cuda") if kwargs.get('device') is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = kwargs.get('batch_size', cfg['GENERAL']['BATCH_SIZE'])
    extractor = kwargs.get('extractor', cfg['GENERAL']['EXTRACTOR'])
    extractor_id = kwargs.get('extractor_id', cfg[extractor]['MODEL_NAME'])
    pretrained = kwargs.get('pretrained', cfg[extractor]['PRETRAINED'])
    dataset_name = kwargs.get('dataset', cfg['GENERAL']['DATASET'])
    split = kwargs.get('split', cfg['GENERAL']['SPLIT'])
    mllm = kwargs.get('mllm', cfg['GENERAL']['MLLM'])
    print(f"Using {extractor} for feature extraction on {dataset_name} ({split})")

    feature_extraction_model, img_preprocess, tokenizer = get_feature_extractor(cfg, 
                                                                                extractor=extractor, 
                                                                                extractor_id=extractor_id, 
                                                                                pretrained=pretrained
                                                                                )
    feature_extraction_model.to(device)
    feature_extraction_model.eval()

    if task.startswith('img2'):
        ### Image Generation Parameters
        image_gen_model_name = kwargs.get('image_gen_model', cfg['GENERAL']['IMAGE_GEN_MODEL']).upper()
        image_size = cfg['IMAGE-GENERATION'][image_gen_model_name]['IMAGE_SIZE']
        n_infer_steps = kwargs.get('n_infer_steps', cfg['IMAGE-GENERATION'][image_gen_model_name]['NUM_INFERENCE_STEPS'])
        image_guidance_scale = kwargs.get('image_guidance_scale', cfg['IMAGE-GENERATION'][image_gen_model_name]['IMAGE_GUIDANCE_SCALE'])
        guidance_scale = kwargs.get('guidance_scale', cfg['IMAGE-GENERATION'][image_gen_model_name]['GUIDANCE_SCALE'])
        use_mllm = kwargs.get('use_mllm', cfg['GENERAL']['USE_MLLM'])

        img_transform_for_generation = transform_image(image_size)

    gen_config, processor, text_generation_model = load_mllm(cfg, **kwargs)
    
    dataloader = get_dataloader(cfg, 
                                split=split.lower(), 
                                mode='relative',
                                dataset_name=dataset_name, 
                                extractor_name=extractor,
                                batch_size=batch_size,
                                transform=img_transform_for_generation if task.startswith('img2') else None
                                )
    
    if task.endswith('2img') and "vit-bigg" not in extractor_id.lower():
        img_batch_size = 2048
    elif task.endswith('2img') and "vit-bigg" in extractor_id.lower():
        img_batch_size = 256
    else:
        img_batch_size = batch_size

    query_feat = []
    query_ids = []
    target_feat = []
    target_ids = []
    target_length = []
    fashioniq_ground_truth = []

    if dataset_name.lower() == 'circo' and split.lower() == 'val':
        gt_img_ids = []
    if dataset_name.lower() == 'cirr' and split.lower() == 'test':
        img_subset = []
        img_subset_ids = []
        img_subset_feat = []
    with torch.no_grad():
        total_time_query = 0.0
        for i, batch in tqdm(enumerate(dataloader), desc=f"Extracting query features for {task}", total=len(dataloader)):
            reference_pil = batch['reference_pil']
            caption = batch['caption']
            query_ids.extend(batch['query_id'])
            if dataset_name.lower() == 'fashioniq':
                fashioniq_ground_truth.extend(batch['target_id'])
            if dataset_name.lower() == 'cirr' and split.lower() == 'test':
                img_subset.extend(batch['image_set'])
                img_subset_ids.extend(batch['image_subset_ids'])
            if dataset_name.lower() == 'circo' and split.lower() == 'val':
                gt_img_ids.extend(batch['target_id'])

            if task.startswith('txt2'):
                composed_messages = list(map(lambda x: get_composed_prompts(dataset_name, *x),zip(reference_pil, caption)))
                if mllm.lower() == 'qwen-3b' or mllm.lower() == 'qwen-7b':
                    composed_messages = {'texts':list(map(lambda x: get_composed_prompts(dataset_name, *x),zip(reference_pil, caption))),
                                             'images':[]
                                        }
                elif mllm.lower() == 'llava' or mllm.lower() == 'deepseek':
                    composed_messages = {'texts':list(map(lambda cap: get_composed_prompts_other_mllm(mllm, cap), caption)),
                                            'images':reference_pil
                                        }
                else:
                    raise ValueError(f"Unsupported MLLM: {mllm}. Should be one of ['QWEN', 'LLAVA', 'DEEPSEEK']")
                start_time_batch = time.time()
                composed_descriptions = generate_texts(composed_messages, gen_config, processor, text_generation_model)
                end_time_batch = time.time()
                total_time_query += end_time_batch - start_time_batch
                # print(composed_descriptions)
                query_feat.append(extract_text_features(composed_descriptions, extractor, tokenizer, feature_extraction_model))
            else:
                raise ValueError(f"Unsupported task: {task}. Should be one of ['txt2img', 'txt2txt', 'img2img', 'img2txt']")
        inference_time_per_sample = total_time_query/len(dataloader.dataset)
        print(f"Finished extracting query features for {task} on {dataset_name}.".upper())
        print("=" * 50)
        print(f"Total time for extracting query features: {total_time_query: .2f} seconds\nPer sample time: {inference_time_per_sample: .2f} seconds\n")
        print("=" * 50)
        
        test_loader = get_test_loader(cfg, dataset_name, split, extractor, img_batch_size)
        start_time_target = time.time()
        if task.endswith('2img'):
            delete_models(text_generation_model)

            for j, test_batch in tqdm(enumerate(test_loader)):
                pil = test_batch['target_pil']
                target_ids.extend(test_batch['target_id'])
                target_length.extend(test_batch['target_length'])

                target_feat.append(extract_image_features(pil, extractor, feature_extraction_model, img_preprocess))

            if dataset_name.lower() == 'cirr' and split.lower() == 'test':
                n_iters = len(img_subset)//img_batch_size + 1 if len(img_subset)%img_batch_size != 0 else len(img_subset)//img_batch_size
                for i in range(n_iters):
                    img_subset_feat.append(extract_image_features(img_subset[i*img_batch_size:(i+1)*img_batch_size], extractor, feature_extraction_model, img_preprocess))
                img_subset_feat = torch.cat(img_subset_feat, dim=0)
                print("Finished extracting subset image features")
                
            delete_models(feature_extraction_model)
        end_time_target = time.time()
        print("Finished extracting target image features")
        print("=" * 50)
        print(f"Total time for extracting target image features: {end_time_target - start_time_target: .2f} seconds\nPer sample time: {(end_time_target - start_time_target)/len(test_loader.dataset): .2f} seconds\n")
        print("=" * 50)

    query_feat = torch.cat(query_feat, dim=0)  
    target_feat = torch.cat(target_feat, dim=0)
    print(query_feat.shape, target_feat.shape)
    
    if dataset_name.lower() == 'circo' and split.lower() == 'test':
        metric = 'map' if dataset_name.lower() == 'circo' else 'recall'
        if dataset_name.lower() == 'circo' and split.lower() == 'val':
            id_to_index = {img_id: idx for idx, img_id in enumerate(target_ids)}
            index= [[id_to_index[i.zfill(12)] for i in group] for group in gt_img_ids]
        else:
            index= None

        res = []
        for k in top_k:
            metric_val = get_metrics(query_feat,
                                      target_feat,
                                      k=k,
                                      target_length=target_length,
                                      metrics=metric,
                                      gt_img_ids=index
                                    )
            print("=" * 50)
            print(f'{metric.upper()}@{k}: {metric_val:.2f}% when using generated description ---> target images\n')
            res.append(metric_val.item())

        avg = sum(res)/len(res)
        print(f"Averge mAP: {avg:.2f}")

        return avg, inference_time_per_sample

def launch(**kwargs):
    cfg = get_default_config("config.yaml")
    seed = kwargs.get('seed', cfg['GENERAL']['SEED'])
    set_seed(seed)
    start = time.time()
    avg_map, inference_time_per_sample, _ = main(cfg, **kwargs)
    end = time.time()
    print("=" * 50)
    print(f"Pipeline time: {end - start:.2f} seconds for seed: {seed}")
    print("=" * 50)
    return avg_map, inference_time_per_sample, end - start


if __name__ == "__main__":
    fire.Fire(launch)