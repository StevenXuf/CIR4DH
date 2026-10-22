# Composed Image Retrieval for Cultural Heritage Collections

Training-free CIR on EUFCC-CIR: Qwen2.5-VL reformulates a reference image and modifier into a target description, then OpenCLIP ranks gallery images by cosine similarity. Experiments evaluate metadata translation and hierarchical information.

![Pipeline](assets/pipeline.png)

## Setup

Use a CUDA-enabled PyTorch environment and run commands from the repository root.

```bash
pip install torch torchvision transformers accelerate open_clip_torch torchmetrics \
    fire omegaconf pandas numpy Pillow tqdm matplotlib wordcloud nltk qwen-vl-utils
mkdir -p logs figures
```

The code unconditionally imports [DeepSeek-VL](https://github.com/deepseek-ai/DeepSeek-VL) and [BLIP](https://github.com/salesforce/BLIP). Install their dependencies and add your BLIP checkout to `PYTHONPATH`. Transformers must support `Qwen2_5_VLForConditionalGeneration`.

## Data

Queries and gallery partitions are provided in `cir_db.csv` and `db_processed.txt`. Use `downloader.py` to download images from URLs in `splits/*.csv`. Its script-level download call is commented out; invoke the function directly:

```bash
python - <<'PYTHON'
from pathlib import Path
import pandas as pd
from downloader import download_images

root = Path("/path/to/GLAM/raw")
for split in Path("splits").glob("*.csv"):
    download_images(pd.read_csv(split), root / split.stem)
PYTHON
```

Set `paths.glam_root` in `config.yaml` to the same root. Images are saved as `<root>/<split>/<idInSource>.png`. The loader filters missing images and duplicate reference/target pairs.

## Evaluation

**`metrics_computation.py` runs the paper's experiments**, reporting Recall@1, @5, @10, @50 (%) and their mean. Defaults use Qwen2.5-VL-7B-Instruct and OpenCLIP ViT-B-16 (`openai` weights).

```bash
python metrics_computation.py --split test_id --gpu_id 0 \
    --baseline gen --use_metadata True --use_translation True \
    --use_first_hierarchy False
```

This configuration uses English-translated metadata with all hierarchies. Repeat with `--split test_ood`. Command-line arguments override `config.yaml`.

| Variant | Arguments |
| --- | --- |
| Catalan metadata | `--use_metadata True --use_translation False` |
| No metadata | `--use_metadata False --use_translation False` |
| First hierarchy branch | `--use_first_hierarchy True` |
| Non-generative baselines | `--baseline text_only`, `image_only`, or `combined` |

Logs append to `logs/`; figures are saved in `figures/`. Preserve logs between translation variants because filenames omit `use_translation`.

**Implementation notes:** `use_first_hierarchy=True` retains the first `$`-separated branch, whereas the paper describes its first label. With `use_metadata=False`, empty metadata can cause word-cloud generation to stop evaluation before metrics are computed.
