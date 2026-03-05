from utils import convert_pil_to_base64

###################################
#Prompts for CIRCO dataset
###################################
def generate_composed_description_qwen(reference_image, caption):
    target_description = [
        {
            "role": "system", 
            "content":
            """
                    You are an expert at visual imagination of real-world scenes. 
                    Given a reference image and modification instructions, mentally apply the modifications and produce an accurate, detailed description of the resulting scene. 
                    Apply the modifications exactly as described, and describe the final scene after the changes.
                    The modifications may involve:
                        1. Cardinality: adjusting the number of objects (e.g., “has two boxes”).
                        2. Addition: introducing new objects or attributes (e.g., “a child under the umbrella”).
                        3. Negation: removing elements (e.g., “shows no bike”).
                        4. Direct Addressing: ensuring a specific object is present (e.g., “next to a window”).
                        5. Compare & Change: altering attributes (e.g., “different color,” “surrounded by flowers”).
                        6. Comparative Statements: relative size, number, or intensity (e.g., “more stickers,” “larger crowd”).
                        7. Conjunction Statements: multiple edits at once (e.g., “surrounded by snow and trees are more bare”).
                        8. Spatial Relations & Background: positioning or environment changes (e.g., “skyscrapers in the background”).
                        9. Viewpoint: changes in perspective or framing (e.g., “shot from above”).
                    Write 1 to 3 coherent sentences in clear English.
                """
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "data:image;base64," + convert_pil_to_base64(reference_image)},
                {
                    "type": "text", 
                    "text": 
                    f"""
                        Here are the modification instructions: {caption}.\n\n
                        Focus on the objects, people, animals, attributes (color, size, shape, quantity), spatial relations, and background context. 
                        Be specific and objective. 
                        Avoid imaginary details not supported by the reference image or the modification. 
                        Do not use vague comparative terms like 'same/different/smaller/larger/shorter/longer/unchanged', etc. Instead, you should specify these differences clearly, like: another color instead of red (if no specific targeting color is mentioned), and a clear sky (if mentioned) instead of unchanged sky, etc.
                        Write 1 to 3 complete and coherent sentences so that I can find targeting images based on your description solely without knowing the reference image or modification instructions.
                        Now, describe how the final image looks after applying these modifications.
                    """
                }
            ],
        }
    ]
    return target_description

def generate_composed_description_llava(caption):
    target_description = [
        {
            "role": "assistant", 
            "content":[{"type": "text", 
            "text":
            """
                    You are an expert at visual imagination of real-world scenes. 
                    Given a reference image and modification instructions, mentally apply the modifications and produce an accurate, detailed description of the resulting scene. 
                    Apply the modifications exactly as described, and describe the final scene after the changes.
                    The modifications may involve:
                        1. Cardinality: adjusting the number of objects (e.g., “has two boxes”).
                        2. Addition: introducing new objects or attributes (e.g., “a child under the umbrella”).
                        3. Negation: removing elements (e.g., “shows no bike”).
                        4. Direct Addressing: ensuring a specific object is present (e.g., “next to a window”).
                        5. Compare & Change: altering attributes (e.g., “different color,” “surrounded by flowers”).
                        6. Comparative Statements: relative size, number, or intensity (e.g., “more stickers,” “larger crowd”).
                        7. Conjunction Statements: multiple edits at once (e.g., “surrounded by snow and trees are more bare”).
                        8. Spatial Relations & Background: positioning or environment changes (e.g., “skyscrapers in the background”).
                        9. Viewpoint: changes in perspective or framing (e.g., “shot from above”).
                    Write 1 to 3 coherent and short sentences in clear English.
                """}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text", 
                    "text": 
                    f"""
                        Here are the modification instructions: {caption}.\n\n
                        Focus on the objects, people, animals, attributes (color, size, shape, quantity), spatial relations, and background context. 
                        Be specific and objective. 
                        Avoid imaginary details not supported by the reference image or the modification. 
                        Do not use vague comparative terms like 'same/different/smaller/larger/shorter/longer/unchanged', etc. Instead, you should specify these differences clearly, like: another color instead of red (if no specific targeting color is mentioned), and a clear sky (if mentioned) instead of unchanged sky, etc.
                        Write 1 to 3 short and coherent sentences so that I can find targeting images based on your description solely without knowing the reference image or modification instructions.
                        Now, describe how the final image looks after applying these modifications.
                    """
                }
            ],
        }
    ]
    return target_description

####################################
# Prompts redirection
####################################
def get_composed_prompts(dataset_name, reference_image, caption):
    if dataset_name.lower() == "circo":
        composed_description = generate_composed_description_qwen(reference_image, caption)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return composed_description


def get_composed_prompts_other_mllm(mllm, caption):
    if mllm.lower() == "llava":
        composed_description = generate_composed_description_llava(caption)
    elif mllm.lower() == "qwen":
        composed_description = generate_composed_description_qwen(caption)
    else:
        raise ValueError(f"Unsupported mllm: {mllm}")
    return composed_description