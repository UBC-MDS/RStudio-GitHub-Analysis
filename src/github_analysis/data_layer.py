import numpy as np
import pandas as pd


def getUniqueProjectIdsFromDf(df):
    return df.project_id.unique()


def getUniqueProjectNamesFromDf(df):
    return df.project_name.unique()


class data_layer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.commits_df = pd.read_feather(data_path)


def getCommitsByProjectName(projectName):
    return(commits_df[commits_df["project_name"] == projectName])


def getCommitsByProjectId(projectId):
    return(commits_df[commits_df["project_id"] == projectId])


def getCommitsByProjectIds(projectIds):
    return commits_df[commits_df.project_id.isin(projectIds)]


def getGroupedCommitsByProjectIds(projectIds):
    projectCommits = getCommitsByProjectIds(projectIds)
    groupedProjects = projectCommits.groupby("project_id")

    return groupedProjects


def getUniqueProjectIds():
    return commits_df.project_id.unique()


def getUniqueProjectNames():
    return commits_df.project_name.unique()


def getUniqueProjectIdsFromDf(df):
    return df.project_id.unique()


def getUniqueProjectNamesFromDf(df):
    return df.project_name.unique()


def getRandomSampleOfIds(n, seed):
    if seed:
        np.random.seed(seed)

    uniqueIds = getUniqueProjectIds()

    return np.random.choice(uniqueIds, n)


def getRandomProjects(n, seed):
    projectIds = getRandomSampleOfIds(n, seed)
    projects = getCommitsByProjectIds(projectIds)

    return projects


if __name__ == '__main__':
    print(getRandomProjects(5, 1))
