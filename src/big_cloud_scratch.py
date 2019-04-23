import pandas_gbq
import networkx as nx
import matplotlib.pyplot as plt

def query_ght(queryString):
    # https://bigquery.cloud.google.com/dataset/ghtorrent-bq:ght
    query_result_df = pandas_gbq.read_gbq(queryString)

    return query_result_df

def plot_commits(commits):
    source_target_commits = commits[["cp_parent_id", "c_id"]].dropna().astype("int64")
    source_target_commits.columns = ["source", "target"]

    g = nx.from_pandas_edgelist(source_target_commits)
    nx.draw_kamada_kawai(g, alpha=0.5, node_color='blue', node_size = 2)

if __name__ == '__main__':
    commitQuery = """
        select
          c.id as c_id,
          p.id as p_id,
          cp.commit_id as cp_commit_id,
          cp.parent_id as cp_parent_id
        from `ghtorrent-bq.ght.commits` c
        left join `ghtorrent-bq.ght.projects` p on (p.id = c.project_id)
        left join `ghtorrent-bq.ght.commit_parents` cp on (cp.commit_id = c.id)
        where (p.id = 12873840)
        limit 10000
    """

    commits = query_ght(commitQuery)
    branchPlot = plot_commits(commits)
    plt.savefig("./imgs/branch_test")
