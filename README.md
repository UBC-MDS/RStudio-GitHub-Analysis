# RStudio-GitHub-Analysis

Contributors: Juno Chen, Ian Flores, Rayce Rossum, Richie Zitomer

Project Mentor: Dr. Tiffany Timbers

Project Partner: Dr. Greg Wilson

## Overview
This project aims to understand how people are currently using GitHub, with the eventual goal of building an easy-to-use alternative to Git.

This project includes the ability to cluster similar GitHub projects and pick out their most commonly-occuring subgraphs.

Motivation behind this project: http://third-bit.com/2017/09/30/git-graphs-and-engineering.html

## Useful documents

- Proposal [Report](https://github.com/UBC-MDS/RStudio-GitHub-Analysis/blob/master/docs/proposal_presentation/proposal_report_final.pdf) and [Presentation](https://github.com/UBC-MDS/RStudio-GitHub-Analysis/blob/master/docs/proposal_presentation/proposal_presentation.html)

- [Final Report](https://github.com/UBC-MDS/RStudio-GitHub-Analysis/blob/master/docs/final_presentation/final_report.pdf) and [Presentation](https://github.com/UBC-MDS/RStudio-GitHub-Analysis/blob/master/docs/final_presentation/final_presentation.html)

- [Blog](https://ubc-mds.github.io/RStudio-GitHub-Analysis/)

## Usage
Run the following commands to reproduce this analysis:
```{bash}
snakemake get_ght_data # Downloads GH Torrent data from figshare. Be aware that the file is quite large, and downloading can take 1-2 hours.

snakemake run_analysis # Run our pipeline; generate embeddings, clusters, tsne graph, motif report, etc.

snakemake generate_images # Generate images of our most important findings.
```

To change parameters from the command line, simply put `--config param=value` after your snakemake call. For a full list of configurable parameters, see the `config.json` file in the root directory of this project. 
For example, if you wanted to run the analysis with 5 workers instead of the default, run:

```{bash}
snakemake run_analysis --config n_workers=5
```

## Config Parameters

|Short Name|Long Name|Description|Default|Type|
|-|-|-|-|-|
| -rp     | --results_path   | The folder to output results of the analysis. e.g. embeddings and plots| ./results/|String|
| -nw     | --n_workers      | The number of workers to use when running the analysis.| 8| int|
| -dp     | --data_path      | The path to the commits.feather file. e.g. /home/user/RStudio-Data-Repository/clean_data/commits_by_org.feather| /home/user/RStudio-Data-Repository/clean_data/commits_by_org.feather|String|
| -np     | --n_projects     | The number of projects to sample from the dataset.| 1000| int|
| -mc     | --min_commits    | The minimum number of commits for a project to be included in the sample.| None | none_or_int|
| -mcount | --min_count      | The min_count parameter for the graph2vec model.| 5| int|
| -nps    | --n_personas     | The number of personas to extract from each cluster.| 5| int|
| -nn     | --n_neurons      | The number of neurons to use for Graph2Vec (project level)| 128| int|
| -ni     | --n_iter         | The number of iteration to use to run the WeisfeilerLehmanMachine| 10| int|
| -rs     | --random_state   | The random state to initalize all random states.| 1| int|


### Example
From the root directory, run: 

```{bash}
python src/python src/github_analysis/main.py -dp "/home/rayce/Assignments/Capstone/RStudio-Data-Repository/clean_data/commits_by_org.feather".py
```

## Installation instructions
First, to get credentials file neccessary for pulling the GitHub Torrent from Google Cloud (necessary for re-generating images for our analysis):

- Follow the instructions under 'Set up a service account' to create and download a credentials file: https://cloud.google.com/video-intelligence/docs/common/auth
- Change the name of the file to `credentials_file.json` and put it in the root directory of the project (a sample file with the name `credentials_file_EXAMPLE.json` is included as a reference).


## Data Repository
[RStudio-Data-Repository](https://github.com/UBC-MDS/RStudio-Data-Repository)
[Figshare Upload](https://figshare.com/articles/GHTorrent_Project_Commits_Dataset/8321285)
