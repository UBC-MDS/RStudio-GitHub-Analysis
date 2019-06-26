import pandas as pd

csv_commits = pd.read_csv('https://storage.cloud.google.com/rstudio_bucket/2019_seed_commits.csv?_ga=2.112003524.-1920784121.1551992733')

csv_commits.to_feather('../artifacts/commits.feather')
