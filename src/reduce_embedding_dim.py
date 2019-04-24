"""Script that takes in an embedding (output of Graph2Vec model), performs t-SNE on it so there are only 2 features,
and then makes a scatterplot with those 2 features as the X- and Y- axes, and outputs that scatterplot to an image file.

This script assumes you're running from the root dir of the project and there is already an embedding in results/ dir
called 'embeddings.csv'
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE


def run_tsne(filename):
    """filename: path to embeddings where first row is the name of the graphs.
    Returns a DataFrame identical to the original but in 2 dimensions."""
    embeddings = pd.read_csv(filename, index_col=0)

    # Make and fit tsne model
    tsne_model = TSNE(n_components=2)
    transformed_array = tsne_model.fit_transform(embeddings.values)
    return pd.DataFrame(transformed_array, columns=['x', 'y'])


def output_image_of_tsne(embeddings_tsne_transform, filename):
    """
    embeddings_tsne_transform: the t-SNE-transformed embedding with 2 columns.
    filename: filepath to output image to.
    """
    fig, ax = plt.subplots()
    ax.scatter(embeddings_tsne_transform.x, embeddings_tsne_transform.y)
    ax.set_title('Embedding Clusters (t-SNE Transformed)')
    plt.savefig(filename)

def reduce():
    transformed_array = run_tsne('./results/embeddings.csv')
    output_image_of_tsne(transformed_array, './results/embeddings_tsne')

if __name__=='__main__':
    reduce()
