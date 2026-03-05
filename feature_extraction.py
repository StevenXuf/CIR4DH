import torch
import torch.nn.functional as F
import open_clip

from tqdm import tqdm
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from torchmetrics.retrieval import RetrievalRecall, RetrievalMAP, RetrievalPrecision
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from open_clip import create_model_from_pretrained, get_tokenizer

from utils import targetpad_transform, transform_image, get_default_config

def get_feature_extractor(cfg, extractor=None, extractor_id=None, pretrained=None):
    if extractor is None:
        extractor = cfg['GENERAL']['EXTRACTOR']
    if extractor_id is None:
        extractor_id = cfg[extractor]['MODEL_NAME']
    if extractor.lower() == 'openclip':
        if pretrained is None:
            pretrained = cfg[extractor]['PRETRAINED']
        print(f"Using pretrained model: {pretrained}")

    if extractor.lower() == 'openvision':
        print(f"Using OpenVision for feature extraction")
        feature_extraction_model, img_preprocess = create_model_from_pretrained(f'hf-hub:{extractor_id}')
        tokenizer = get_tokenizer(f'hf-hub:{extractor_id}')
    elif extractor.lower() == 'openclip':
        print(f"Using OpenCLIP for feature extraction")
        feature_extraction_model, _, img_preprocess = open_clip.create_model_and_transforms(extractor_id, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(extractor_id)
    elif extractor.lower() == 'clip':
        print(f"Using CLIP for feature extraction")
        feature_extraction_model = AutoModel.from_pretrained(extractor_id)
        tokenizer = AutoTokenizer.from_pretrained(extractor_id)
        img_preprocess = targetpad_transform(cfg[extractor]['IMAGE_MEAN'], cfg[extractor]['IMAGE_STD'], target_ratio=1.25, dim=cfg[extractor]['IMAGE_SIZE'])
    elif extractor.lower() == 'siglip2':
        print(f"Using SigLip2 for feature extraction")
        feature_extraction_model = AutoModel.from_pretrained(extractor_id)
        tokenizer = AutoTokenizer.from_pretrained(extractor_id)
        img_preprocess = AutoProcessor.from_pretrained(extractor_id)
    else:
        raise ValueError(f"Unknown extractor: {extractor}")
    return feature_extraction_model, img_preprocess, tokenizer

def get_metrics(feat1, feat2, k, target_length, metrics='recall', gt_img_ids=None):
    if metrics == 'recall':
        compute = RetrievalRecall(top_k=k)
    elif metrics == 'precision':
        compute = RetrievalPrecision(top_k=k)
    elif metrics == 'map':
        compute = RetrievalMAP(top_k=k)

    sim = pairwise_cosine_similarity(feat1, feat2)
    indexes = torch.arange(sim.size(0), dtype=torch.long).unsqueeze(1).expand(*sim.size()).to(sim.device)

    targets = torch.zeros(sim.size(0), sim.size(1), dtype=torch.long).to(sim.device)
    if metrics == 'map' and gt_img_ids is not None:
        for i in range(sim.size(0)):
           targets[i, gt_img_ids[i]] = 1
    else:
        start = 0
        for i, length in enumerate(target_length):
            targets[i, start:start+length] = 1
            start += length

    res = compute(sim,targets,indexes=indexes)

    _, ids = sim.topk(k, dim=1)
    return res*100

def extract_features(model,processor,dataloader,config_path='./config.yaml',cfg=None):

    if cfg is None:
        cfg=get_default_config(config_path)
    
    device=torch.device(f'cuda:{cfg["GENERAL"]["DEVICE"]}' if torch.cuda.is_available() else 'cpu')

    print(f'Using {cfg["CLIP"]["MODEL_NAME"].upper()} for feature extraction')
    model.to(device)

    ref_image_features = []
    candidate_image_features = []
    text_features = []
    with torch.no_grad():
        print(f"Processing batches...")
        for batch in tqdm(dataloader):
            target_images = batch['target'].to(device)
            reference_images = batch['reference'].to(device)
            captions = batch['caption']

            text_inputs = processor(text=captions, return_tensors="pt", padding=True, truncation=True).to(device)

            text_feat = model.get_text_features(**text_inputs)
            ref_feat = model.get_image_features(pixel_values=reference_images)
            candidate_feat = model.get_image_features(pixel_values=target_images)

            ref_image_features.append(ref_feat)
            text_features.append(text_feat)
            candidate_image_features.append(candidate_feat)
        print(f"Finished processing batches.")

    ref_image_features = torch.cat(ref_image_features, dim=0)
    candidate_image_features = torch.cat(candidate_image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)

    return ref_image_features, candidate_image_features, text_features

def perform_retrieval(ref_image_features, candidate_image_features, text_features, TOP_K):

    ##################feature based retrival####################
    print('\n' + '='*50 + '\nFeature-based Image Retrieval\n' + '='*50)
    recall_ref2can=get_metrics(ref_image_features,candidate_image_features,TOP_K)
    print(f'Recall@{TOP_K}: {recall_ref2can:.2f} when using image-to-image retrieval')
    recall_t2ref=get_metrics(text_features,candidate_image_features,TOP_K)
    print(f'Recall@{TOP_K}: {recall_t2ref:.2f} when using text-to-image retrieval')

    #image+text---->image
    print(f'Using image+text pairs to retrieve image')
    added_features=F.normalize(ref_image_features+text_features,p=2,dim=1)
    recall_added=get_metrics(added_features,candidate_image_features,TOP_K)
    print(f'Recall@{TOP_K}: {recall_added:.2f} when using add(image,text)---->image')

    #image*text---->image
    print(f'Using image*text pairs to retrieve image')
    multiplied_features=F.normalize(ref_image_features*text_features,p=2,dim=1)
    recall_multiplied=get_metrics(multiplied_features,candidate_image_features,TOP_K)
    print(f'Recall@{TOP_K}: {recall_multiplied:.2f} when using multiply(image,text)---->image')

def main(cfg):
    MODEL_FILENAME = cfg['CLIP']['MODEL_NAME']
    model = AutoModel.from_pretrained(MODEL_FILENAME)
    processor = AutoProcessor.from_pretrained(MODEL_FILENAME)
    mean = cfg['CLIP']['IMAGE_MEAN']
    std = cfg['CLIP']['IMAGE_STD']
    img_transform = transform_image(cfg['CLIP']['IMAGE_SIZE'], mean, std)
    dataloader = get_refined_fashioniq_loader(cfg['RefinedFashionIQ']['OUTPUT_DIR'], transform=img_transform, batch_size=cfg['GENERAL']['BATCH_SIZE'])
    ref_image_features, candidate_image_features, text_features = extract_features(model, processor, dataloader=dataloader, cfg=cfg)
    perform_retrieval(ref_image_features, candidate_image_features, text_features, cfg['GENERAL']['TOP_K'])

if __name__=='__main__':
    cfg= get_default_config("config.yaml")
    torch.manual_seed(cfg['GENERAL']['SEED'])
    main(cfg)
    # torch.manual_seed(42)
    # feat1 = torch.randn(5, 512)
    # feat2 = torch.randn(10, 512)
    # length = [1,2,2,3,2]
    # res = get_metrics(feat1, feat2, 5, length)
    # print(res)