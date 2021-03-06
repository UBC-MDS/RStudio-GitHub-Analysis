"""
Script that takes in an embedding (output of Graph2Vec model), performs t-SNE on it so there are only 2 features,
and then makes a scatterplot with those 2 features as the X- and Y- axes, and outputs that scatterplot to an image file.

This script assumes you're running from the root dir of the project and there is already an embedding in results/ dir
called 'embeddings.csv'
"""

import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE

def run_tsne(workers=4, embeddings=None, filename=None, random_state=None):
    """ Fit 2-D T-SNE model on input embeddings.

        Parameters
        ----------
        embedding: embeddings to run T-SNE on.
        filename: path to embeddings where first row is the name of the graphs.
        random_state: can be specified to allow reproduciblity.

        Returns
        -------
        A DataFrame identical to the original but in 2 dimensions.
    """
    if embeddings is None and filename is not None:
        embeddings = pd.read_csv(filename, index_col=0)

    # Make and fit tsne model
    tsne_model = TSNE(n_components=2, n_jobs=workers, random_state=random_state)
    transformed_array = tsne_model.fit_transform(embeddings.values)
    df = pd.DataFrame(transformed_array, columns=['x', 'y'])
    df.index = embeddings.index
    return df

def output_image_of_tsne(embeddings_tsne_transform, filename):
    """ Output the T-SNE embeddings into a scatterplot.

        Parameters
        ----------
        embeddings_tsne_transform: the t-SNE-transformed embedding with 2 columns.
        filename: filepath to output image to.

        Returns
        -------
        An image of 2-D T-SNE.
    """
    fig, ax = plt.subplots()
    ax.scatter(embeddings_tsne_transform.x, embeddings_tsne_transform.y)
    ax.set_title('Embedding Clusters (t-SNE Transformed)')
    plt.savefig(filename)


def reduce_dim(output_path, input_path, workers=4, embeddings=None, save_to_csv=True, save_image=True, random_state=None):
    transformed_array = run_tsne(workers=workers, filename=input_path, random_state=random_state)

    if save_to_csv == True:
        transformed_array.to_csv(output_path + 'embeddings_reduced_dim.csv')

    if save_image == True:
        output_image_of_tsne(transformed_array, output_path + 'embeddings_tsne')

if __name__=='__main__':
    reduce_dim()
