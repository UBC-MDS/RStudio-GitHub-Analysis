def git_graph(commitData):
    """
    Function to generate the commit graph in networkx from the query results.
    :param commitData: Data pulled from the commit_query string.
    :return nxGraph: Networkx graph
    """
    source_target_commits = commitData[["parent_id", "commit_id"]].dropna().astype("int64")
    source_target_commits.columns = ["source", "target"]

    return nx.from_pandas_edgelist(source_target_commits, create_using=nx.OrderedDiGraph())

def multicore_git_graph(d, projectId, projectGraph):
    d[projectId] = git_graph(projectGraph)
