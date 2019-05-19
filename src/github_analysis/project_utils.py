import nxutils
import data_layer as dl

def git_graph_from_project_id(projectId):
    return nxutils.git_graph(dl.getCommitsByProjectId(projectId))
