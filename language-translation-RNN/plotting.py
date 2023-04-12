# this is what it took to get matplotlib working on a macbook.
# you may have to tweak this to get the matplotlib imports working on your own machine.
# let us know if you have problems.  -- the TAs

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker
from typing import List
import numpy as np


def visualize_attention(source_sentence_str: List[str],
                        target_sentence_str: List[str],
                        attention_matrix: np.ndarray,
                        outfile: str):
    """

    :param source_sentence_str: the source sentence, as a list of strings
    :param target_sentence_str: the target sentence, as a list of strings
    :param attention_matrix: the attention matrix, of dimension [target_sentence_len x source_sentence_len]
    :param outfile: the file to output to
    """
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attention_matrix, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + source_sentence_str, rotation=90)
    ax.set_yticklabels([''] + target_sentence_str)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(outfile)

    plt.close()
