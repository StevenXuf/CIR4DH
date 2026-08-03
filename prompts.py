from utils import convert_pil_to_base64

def get_sys_prompt(pil_image, modifier, gen_model='qwen', material=None, obj=None):
    meta_info = f"""
                Additionally, the given reference image contains hierarchical information for objects: "{obj}" and for material: "{material}" in Cantalan language.
                You shold first translate the Catalan information into English in your mind, and then take into account the translated material and object information of the reference image when generating the final description of target images.
                """

    template = """
                You are an expert in visual reasoning for composed image retrieval.

                Given a reference image and a modification instruction, imagine the scene that would result after applying the instruction to the reference image.

                Follow the steps below:

                1. Identify the visually important objects, and their attributes in the reference image.
                2. Apply the requested modification exactly to the targeting object. Replace any original detail that conflicts with the modification.
                3. Preserve all relevant details from the reference image that are not changed by the instruction.
                4. Make the modified content explicit, especially changes involving objects, or materials.
                5. Do not mention the reference image, the modification instruction, or the editing process.
                6. Do not describe both the original and modified states.
                7. Do not add details that are not visible or reasonably implied.
                8. Output two to three simple, concise, coherent, and visually descriptive English sentences for the resulting images. 

                For example, if the reference image shows: "a ceramic on a wooden table in a kitchen" and the instruction is "change ceramic to brick", then retuen only: "a brick on a wooden table. The table is in a kitchen full of furniture".
            """
    if gen_model == 'qwen':
        sys_prompt = [
            {
                "role": "system", 
                "content": f"{template}"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": "data:image;base64," + convert_pil_to_base64(pil_image)},
                    {   
                        "type": "text",
                        "text": f"""
                                Modification: {modifier}
                                {meta_info}
                                Now give the final caption in two to three short sentences.
                            """ if material or obj else f"""
                                Modification: {modifier}
                                Now give the final caption in two to three short sentences.
                            """
                    }
                ],
            }
        ]
    elif gen_model == 'llava':
        sys_prompt = [
            {
                "role": "system", 
                "content": [
                    {
                        "type": "text",
                        "text": f"{template}"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {   
                        "type": "text",
                        "text": f"""
                                Modification: {modifier}
                                {meta_info}
                                Now give the final caption in two to three short sentences.
                            """ if material or obj else f"""
                                Modification: {modifier}
                                Now give the final caption in two to three short sentences.
                            """
                    },
                    {"type": "image"}
                ],
            }
        ]
    elif gen_model == 'deepseek':
        sys_prompt = [
            {
                "role": "User",
                "content": f"""{template}
                                Modification: {modifier}
                                {meta_info}
                                Now give the final caption in two to three short sentences.
                            """ if material or obj else f"""{template}
                                Modification: {modifier}
                                Now give the final caption in two to three short sentences.""",
                "images": [],
            },
            {"role": "Assistant", "content": ""},
        ]
    else:
        raise ValueError(f"Unsupported gen_model: {gen_model}")

    return sys_prompt