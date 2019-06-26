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
PLACEHOLDER 

### Example
From the root directory, run: 

```{bash}
python src/python src/github_analysis/main.py -dp "/home/rayce/Assignments/Capstone/RStudio-Data-Repository/clean_data/commits_by_org.feather".py
```

## Installation instructions
To get credentials file for GitHub Torrent Google Cloud (necessary for re-running the pipeline to generate images):

- Follow the instructions here to create and download a credentials file: https://developers.google.com/adwords/api/docs/guides/authentication#generate_oauth2_credentials
- Change the name of the file to `credentials_file.json` and put it in the root directory of the project (a sample file with the same name is included as a reference).


## Data Repository
[RStudio-Data-Repository](https://github.com/UBC-MDS/RStudio-Data-Repository)
