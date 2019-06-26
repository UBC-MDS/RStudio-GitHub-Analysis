FROM continuumio/anaconda3

RUN apt-get install -y cmake

RUN conda install --quiet --yes -c conda-forge  \
  multicore-tsne \
  pandas-gbq \
  panel \
  networkx \
  joblib \
  gensim \
  tqdm

RUN conda install --quiet --yes -c pyviz pyviz

RUN conda install --quiet --yes -c bioconda snakemake
