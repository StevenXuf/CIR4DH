import torch
import fire
import logging
from transformers import Qwen2_5_VLForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoModelForCausalLM,  AutoProcessor, GenerationConfig, set_seed
from qwen_vl_utils import process_vision_info
from deepseek_vl.models import VLChatProcessor

from prompts import get_sys_prompt
from utils import get_config, setup_logging

def load_text_gen_model(cfg, logger, **kwargs):
    gen_model = kwargs.get('gen_model', cfg.gen_model)
    temperature = kwargs.get('temperature', cfg.qwen.temperature)
    top_p = kwargs.get('top_p', cfg.qwen.top_p)
    llm_top_k = kwargs.get('top_k', cfg.qwen.top_k)
    max_new_tokens = kwargs.get('max_new_tokens', cfg.qwen.max_new_tokens)
    device = torch.device(f"cuda:{kwargs.get('gpu_id', cfg.gpu_id)}" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading {gen_model.upper()} model on device: {device}")
    logger.info(f"Generation parameters: temperature={temperature}, top_p={top_p}, top_k={llm_top_k}, max_new_tokens={max_new_tokens}")

    gen_config = GenerationConfig(do_sample=True,
                                    temperature=temperature,
                                    top_p=top_p,
                                    top_k=llm_top_k,
                                    max_new_tokens=max_new_tokens,
                                )
    
    if gen_model == "llava":
        model_id = cfg.llava.model_id
        text_generation_model = LlavaNextForConditionalGeneration.from_pretrained(model_id, 
                                                                            dtype=torch.bfloat16, 
                                                                            device_map={"": device}
                                                                            ).to(device)
    elif gen_model == "deepseek":
        model_id = cfg.deepseek.model_id
        text_generation_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        text_generation_model = text_generation_model.to(torch.bfloat16).to(device)
    elif gen_model == "qwen":
        model_id = cfg.qwen.model_id
        text_generation_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, 
                                                                                dtype=torch.bfloat16, 
                                                                                device_map={"": device}
                                                                                ).to(device)
    else:
        logger.error(f"Unsupported MLLM: {gen_model.upper()}. Should be one of ['QWEN', 'LLAVA', 'DEEPSEEK']")
        raise ValueError(f"Unsupported MLLM: {gen_model.upper()}. Should be one of ['QWEN', 'LLAVA', 'DEEPSEEK']")
    
    text_generation_model.eval()
    
    if gen_model == "deepseek":
        processor = VLChatProcessor.from_pretrained(model_id)
    elif gen_model == "llava":
        processor = LlavaNextProcessor.from_pretrained(model_id, 
                                                padding_side='left', 
                                                use_fast=True
                                                )
    else:
        processor = AutoProcessor.from_pretrained(model_id, 
                                                padding_side='left', 
                                                use_fast=True
                                                )
    return text_generation_model, processor, gen_config

@torch.inference_mode()
def generate_text(cfg, logger, pil_images, modifiers, model, processor, gen_config, **kwargs) -> list[str]:
    gen_model = cfg.gen_model
    if cfg.use_metadata and ('materials' in kwargs and 'objects' in kwargs):
        prompts = [get_sys_prompt(pil_image, modifier, gen_model, material, obj) for pil_image, modifier, material, obj in zip(pil_images, modifiers, kwargs['materials'], kwargs['objects'])]
    else:
        prompts = [get_sys_prompt(pil_image, modifier, gen_model) for pil_image, modifier in zip(pil_images, modifiers)]
    
    if gen_model == "qwen":
        texts = [
            processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts
        ]
        image_inputs, video_inputs = process_vision_info(prompts)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            padding_side='left'
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, generation_config=gen_config)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    elif gen_model == "llava":
        texts = [
            processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts
        ]
        inputs = processor(
            text=texts,
            images=pil_images,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(**inputs, generation_config=gen_config)
        output_texts = processor.batch_decode(generated_ids[:,inputs["input_ids"].size(-1):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
    elif gen_model == "deepseek":
        tokenizer = processor.tokenizer
        info = [processor.process_one(
            conversations=prompt,
            images=[pil_image],
        ) for prompt, pil_image in zip(prompts, pil_images)]

        inputs = processor.batchify(info).to(model.device)
        inputs_embeds = model.prepare_inputs_embeds(**inputs)

        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            top_k=gen_config.top_k,
            max_new_tokens=gen_config.max_new_tokens,
        )

        output_texts = [tokenizer.decode(outputs[i].cpu().tolist(), skip_special_tokens=True) for i in range(len(outputs))]
    else:
        logger.error(f"Unsupported MLLM: {gen_model.upper()}. Should be one of ['QWEN', 'LLAVA', 'DEEPSEEK']")
        raise ValueError(f"Unsupported MLLM: {gen_model.upper()}. Should be one of ['QWEN', 'LLAVA', 'DEEPSEEK']")

    logger.info(f"Generated texts: {output_texts}")
    return output_texts

def main(**kwargs):
    cfg = get_config(**kwargs)
    set_seed(cfg.seed)
    setup_logging(cfg.paths.log_path)
    logger = logging.getLogger(__name__)
    load_text_gen_model(cfg, logger, **kwargs)


if __name__ == "__main__":
    fire.Fire(main)