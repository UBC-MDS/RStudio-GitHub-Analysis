import feather
import numpy as np

commitDataPath = "/Users/richiezitomer/Documents/RStudio-Data-Repository/clean_data/commits.feather"
commits_df = feather.read_dataframe(commitDataPath)

def getCommitsByProjectName(projectName):
    return(commits_df[commits_df["project_name"] == projectName])

def getCommitsByProjectId(projectId):
    return(commits_df[commits_df["project_id"] == projectId])

def getUniqueProjectIds():
    return commits_df.project_id.unique()

def getUniqueProjectNames():
    return commits_df.project_name.unique()

def getRandomSampleOfIds(n, seed):
    if seed:
        np.random.seed(seed)

    uniqueIds = getUniqueProjectIds()

    return np.random.choice(uniqueIds, n)

def getRandomProjects(n, seed):
    projectIds = getRandomSampleOfIds(n, seed)

    projects = {}
    for id in projectIds:
        projects[id] = getCommitsByProjectId(id)

    return projects

def getCommitsByProjectIds(projectIds):
    return commits_df[commits_df.project_id.isin(projectIds)]

if __name__ == '__main__':
    print(getRandomProjects(5, 1))
