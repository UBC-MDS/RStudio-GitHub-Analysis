import pandas as pd
import numpy as np

def getUniqueProjectIdsFromDf(df):
    return df.project_id.unique()

def getUniqueProjectNamesFromDf(df):
    return df.project_name.unique()

class data_layer:
    def __init__(self, data_path, min_number_commits=None):
        self.data_path = data_path
        self.commits_df = pd.read_feather(data_path)
        if min_number_commits is not None:
            grouped_projects = self.commits_df.groupby('project_id')
            self.commits_df = grouped_projects.filter(lambda x: x.commit_id.count() > min_number_commits)

    def getProjectsDf(self):
        return self.commits_df

    def getCommitsByProjectName(self, project_name):
        return self.commits_df[self.commits_df["project_name"] == project_name]

    def getCommitsByProjectId(self, project_id):
        return self.commits_df[self.commits_df["project_id"] == project_id]

    def getCommitsByProjectIds(self, projects_ids):
        return self.commits_df[self.commits_df.project_id.isin(projects_ids)]

    def getGroupedCommitsByProjectIds(self, projects_ids):
        projectCommits = self.getCommitsByProjectIds(projects_ids)
        groupedProjects = projectCommits.groupby("project_id")

        return groupedProjects

    def getUniqueProjectIds(self):
        return self.commits_df.project_id.unique()

    def getUniqueProjectNames(self):
        return self.commits_df.project_name.unique()

    def getRandomSampleOfIds(self, n, seed):
        if seed:
            np.random.seed(seed)

        uniqueIds = self.getUniqueProjectIds()

        return np.random.choice(uniqueIds, n, replace=False)

    def getRandomProjects(self, n, seed):
        projectIds = self.getRandomSampleOfIds(n, seed)
        projects = self.getCommitsByProjectIds(projectIds)

        return projects

    def getProjectNameById(self, project_id):
        return self.commits_df[self.commits_df["project_id"] == project_id].project_name.unique()[0]

if __name__ == '__main__':
    print(getRandomProjects(5, 1))
