import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_gbq
import github_analysis.data_layer as dl
import github_analysis.motif_finder as mf
from google.oauth2.service_account import Credentials
from sklearn import preprocessing


def pull_queries(query_string, google_project_id='planar-elevator-238518', credentials_file='My Project 15448-4edea2614a66.json'):
    credentials = Credentials.from_service_account_file(credentials_file)
    return pandas_gbq.read_gbq(query_string,project_id=google_project_id,credentials=credentials)


def main(data_path='/Users/richiezitomer/Documents/RStudio-Data-Repository/clean_data/commits.feather'):
    emb = pd.read_csv('results/embeddings.csv')
    project_ids = emb.type.values
    proj_ids_string = ",".join(project_ids.astype(str))

    #Load data
    pickle_in = open("./results/motifs_by_cluster.pickle","rb")
    motifs_by_cluster = pickle.load(pickle_in)

    pickle_in = open("./results/clusters.pickle","rb")
    clusters = pickle.load(pickle_in)

    # Load Data
    comm_auth_by_proj = pull_queries(COMM_AUTH_BY_PROJ.format(proj_ids = proj_ids_string)).set_index('p_id') #pd.read_csv('data/author_commits_by_proj_100.csv').set_index('p_id')
    pr_cr_by_proj = pull_queries(PR_CR_BY_PROJ.format(proj_ids = proj_ids_string)).set_index('p_id') #pd.read_csv('data/pr_cr_by_proj_100.csv').set_index('p_id')
    issues_by_proj = pull_queries(ISSUES_BY_PROJ.format(proj_ids = proj_ids_string)).set_index('p_id') #pd.read_csv('data/issues_by_proj_100.csv').set_index('p_id')
    owner_age_by_proj = pull_queries(OWNER_AGE_BY_PROJ.format(proj_ids = proj_ids_string)).set_index('p_id') #pd.read_csv('data/owner_age_by_proj_100.csv').set_index('p_id')
    time_betw_commits_by_proj = pull_queries(TBC_BY_PROJ.format(proj_ids = proj_ids_string)).set_index('p_id') #pd.read_csv('data/time_between_commits_100.csv').set_index('p_id')[['mean_tbc']]

    project = pd.concat([comm_auth_by_proj,pr_cr_by_proj,issues_by_proj,owner_age_by_proj,time_betw_commits_by_proj],axis=1)
    #project = pd.concat([comm_auth_by_proj,pr_cr_by_proj,issues_by_proj],axis=1)

    cluster_lookup = {}
    for cluster,value in clusters.items():
        for proj in value:
            cluster_lookup[proj] = cluster

    project['cluster'] = project.reset_index().p_id.apply(lambda x: cluster_lookup[x]).values

    # project['owner_age'] = project.owner_age/30

#    multi_chain_perc = 100-pd.Series([79.2,71.8,91.1,60.3,83,96.6,83.6,55.4,81.5,96.6])
#    multi_chain_perc.name='complexity'

    # Should make a function out of this instead of doing it a bunch of times
    commits_dl = dl.data_layer(data_path)
    motifs_by_cluster = mf.get_motifs_by_cluster(clusters, commits_dl,k_for_motifs=5,
                                                 output_file=None)

    multi_chain_perc_5 = []
    for cluster in sorted(motifs_by_cluster.keys()):
        for motif in motifs_by_cluster[cluster]:
            if sum([motif.out_degree(node) in [0, 1] for node in motif]) == len(motif.nodes):
                multi_chain_perc_5.append(1000-motifs_by_cluster[cluster][motif])
    multi_chain_perc_5 = pd.Series(multi_chain_perc_5)
    multi_chain_perc_5.name = 'complexity_5'

    project = pd.merge(project,multi_chain_perc_5,left_on='cluster',right_index=True)

    motifs_by_cluster = mf.get_motifs_by_cluster(clusters, commits_dl,k_for_motifs=25,
                                                 output_file=None)

    multi_chain_perc_25 = []
    for cluster in sorted(motifs_by_cluster.keys()):
        for motif in motifs_by_cluster[cluster]:
            if sum([motif.out_degree(node) in [0, 1] for node in motif]) == len(motif.nodes):
                multi_chain_perc_25.append(1000-motifs_by_cluster[cluster][motif])
        multi_chain_perc_25 = pd.Series(multi_chain_perc_25)
        multi_chain_perc_25.name = 'complexity_25'

    project = pd.merge(project,multi_chain_perc_25,left_on='cluster',right_index=True)

    motifs_by_cluster = mf.get_motifs_by_cluster(clusters, commits_dl,k_for_motifs=100,
                                                 output_file=None)

    multi_chain_perc_100 = []
    for cluster in sorted(motifs_by_cluster.keys()):
        for motif in motifs_by_cluster[cluster]:
            if sum([motif.out_degree(node) in [0, 1] for node in motif]) == len(motif.nodes):
                multi_chain_perc_100.append(1000-motifs_by_cluster[cluster][motif])
    multi_chain_perc_100 = pd.Series(multi_chain_perc_100)
    multi_chain_perc_100.name = 'complexity_100'

    project = pd.merge(project,multi_chain_perc_100,left_on='cluster',right_index=True)


    # proj_no_commits = project.drop('commits',axis=1)
    # proj_no_commits['commits/10'] = project.commits/10

    # Normalize/Standardize
    # Get column names first
    names = project.columns
    # Create the Scaler object
    scaler = preprocessing.StandardScaler()
    # Fit your data on the scaler object
    scaled_df = scaler.fit_transform(project)
    scaled_df = pd.DataFrame(scaled_df, columns=names)

    sns.clustermap(project.groupby('cluster').mean(),cmap='OrRd')
    plt.savefig('./results/Report_VO.png')




PR_CR_BY_PROJ = """SELECT 
          p.id as p_id, 
      count(distinct pr.id) as prs,
      count(distinct prc.comment_id) as code_reviews
    FROM `ghtorrent-bq.ght.projects` p
    left join `ghtorrent-bq.ght.pull_requests` pr on (pr.head_repo_id=p.id)
    left join `ghtorrent-bq.ght.pull_request_comments` prc on (prc.pull_request_id=pr.id)
    where p.id in ({proj_ids})
    group by p.id
    """

COMM_AUTH_BY_PROJ = """SELECT 
      p.id as p_id, 
      count(distinct c.id) as commits,
      count(distinct c.author_id) as authors
    FROM `ghtorrent-bq.ght.projects` p
    left join `ghtorrent-bq.ght.commits` c on (c.project_id=p.id)
    where p.id in ({proj_ids})
    group by p.id
    """

ISSUES_BY_PROJ = """SELECT 
      p.id as p_id, 
     count(distinct i.id) as issues
    FROM `ghtorrent-bq.ght.projects` p
    left join `ghtorrent-bq.ght.issues` i on (i.repo_id=p.id)
    where p.id in ({proj_ids})
    group by p.id
    """

OWNER_AGE_BY_PROJ = """SELECT
    p.id AS p_id,
    date_diff(DATE(p.created_at), DATE(u.created_at), DAY)/30 as owner_age
    FROM `ghtorrent-bq.ght.projects` p 
    left join `ghtorrent-bq.ght.users` u on (u.id = p.owner_id)
    where p.id in ({proj_ids})
    and date_diff(DATE(p.created_at), DATE(u.created_at),DAY)>=0"""

TBC_BY_PROJ = """select 
  project_id as p_id,
  avg(date_diff(date(ca),date(ca_lag),DAY)) as mean_tbc
from (SELECT 
          created_at as ca,
          lag(created_at,1) over (partition by project_id order by created_at) as ca_lag,
          project_id
      FROM `ghtorrent-bq.ght.commits`
      where project_id in ({proj_ids})
        and EXTRACT(YEAR FROM created_at)>=2011
        and EXTRACT(YEAR FROM created_at)<=2016
        order by created_at)
group by project_id
"""

if __name__ == '__main__':
    main()
