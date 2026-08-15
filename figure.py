import fire
import logging
import nltk
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords as nltk_stopwords

from pathlib import Path
from transformers import set_seed

from utils import get_config, setup_logging
from loading_data import GLAMQueryDataset

nltk.download("stopwords")

'''
python figure.py --split test_id --seed 42
'''
def plot_samples(**kwargs):
    cfg = get_config(**kwargs)
    set_seed(cfg.seed)

    path = Path(cfg.paths.log_path) / f"{cfg.split}_sample_{cfg.seed}.log"
    setup_logging(str(path))
    logger = logging.getLogger(__name__)

    logger.info("Loading the query dataset.")
    query_dataset = GLAMQueryDataset(cfg, logger).load_dataset()

    sample_size = 5
    cols = 3  # Reference Image, Modifier, Target Image
    fig, axes = plt.subplots(sample_size, cols, figsize=(5 * cols, 5 * sample_size))

    indices = np.random.choice(len(query_dataset), sample_size, replace=False)
    for j in range(sample_size):
        idx = indices[j]
        logger.info(f"Selected sample at index {idx}.")
        sample = query_dataset[idx]
        ref_pil = sample['pil_image']
        ref_path = sample['ref_path']
        target_pil = sample['target_pil']
        target_paths = sample['target_paths']
        modifier = sample['modifier']

        logger.info(f"Reference image path of sample {idx}: {ref_path}")
        logger.info(f"Target image paths of sample {idx}: {target_paths}")

        # Display the images and their corresponding information
        for i in range(cols):
            axes[j, i].axis('off')
            if i == 0:
                axes[j, i].imshow(ref_pil)
                if j == 0:
                    axes[j, i].set_title("Reference Image", fontsize=20, fontweight='bold')
            elif i == 1:
                text = "\n".join(wrap(modifier, width=25))
                axes[j, i].text(0.5, 0.52, text, 
                            fontsize=18, fontweight='bold', ha='center', va='center')
                axes[j, i].annotate("",
                                    xy=(0.95, 0.42),
                                    xytext=(0.05, 0.42),
                                    xycoords="axes fraction",
                                    arrowprops=dict(
                                        arrowstyle="-|>",
                                        color="#b5adaf",
                                        lw=4,
                                        mutation_scale=22,
                                        shrinkA=0,
                                        shrinkB=0,
                                    )
                )
                if j == 0:
                    axes[j, i].set_title("Modifier", fontsize=20, fontweight='bold')
            elif i == 2:
                axes[j, i].imshow(target_pil[0])
                if j == 0:
                    axes[j, i].set_title("Target Image", fontsize=20, fontweight='bold')

    plt.tight_layout()

    save_path = Path(cfg.paths.figure_path) / f"{cfg.split}_sample_{cfg.seed}.pdf"
    plt.savefig(save_path, bbox_inches='tight')
    logger.info(f"Sample plot saved to {save_path}")


def plot_wordcloud(
    sentences,
    file_name,
    custom_stopwords=None,
    width=1200,
    height=600,
    figsize=(5,4),
):
    """
    Plot a word cloud from a list of English sentences.

    Args:
        sentences (list[str]): List of input sentences.
        custom_stopwords (list[str] | set[str] | None):
            Additional words to remove.
        width (int): Word cloud width.
        height (int): Word cloud height.
        figsize (tuple): Matplotlib figure size.
    """
    # Combine sentences
    text = " ".join(sentences)

    # English stopwords + custom stopwords
    stopwords = set(STOPWORDS).union(set(nltk_stopwords.words('catalan')))

    if custom_stopwords:
        stopwords.update(word.lower() for word in custom_stopwords)

    # Generate word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color="white",
        stopwords=stopwords,
        collocations=False,
    ).generate(text)

    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"./figures/{file_name}.pdf", bbox_inches='tight')

if __name__ == "__main__":
    fire.Fire(plot_samples)