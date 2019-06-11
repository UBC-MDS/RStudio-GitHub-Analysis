import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas_gbq
import github_analysis.data_layer as dl
import github_analysis.motif_finder as mf
from google.oauth2.service_account import Credentials
from sklearn import preprocessing


def pull_queries(query_string, google_project_id='planar-elevator-238518', credentials_file='My Project 15448-4edea2614a66.json'):
    credentials = Credentials.from_service_account_file(credentials_file)
    return pandas_gbq.read_gbq(query_string,project_id=google_project_id,credentials=credentials)


class Heatmapper:
    def __init__(self, num_motifs_to_sample=100, motif_lengths=[5,25,50,100], data_path='/Users/richiezitomer/Documents/RStudio-Data-Repository/clean_data/commits.feather',
                 embedding_path='results/embeddings.csv', cluster_path="./results/clusters.pickle"):
        self.data_path = data_path
        self.commits_dl = dl.data_layer(data_path)
        # self.motifs_generated = False
        self.num_motifs_to_sample = num_motifs_to_sample
        self.motif_lengths = motif_lengths
        self.emb = pd.read_csv(embedding_path)

        project_ids = self.emb.type.values
        self.proj_ids_string = ",".join(project_ids.astype(str))

        pickle_in = open(cluster_path, "rb")
        self.clusters = pickle.load(pickle_in)

    def generate_motifs(self, k, read_from_file=False,output_file_prefix='results/motifs_by_cluster_{}.pickle'):
        """Generate ."""
        # for k in self.motif_lengths:
        output_file = output_file_prefix.format(str(k))
        if read_from_file:
            if os.path.isfile(output_file):
                print('{}-Length Motif by Cluster Already Exists, so using that. Set read_from_file=False if you want to regenerate it.'.format(k))
                pickle_in = open(output_file, "rb")
                return pickle.load(pickle_in)
            else:
                motifs_by_cluster = mf.get_motifs_by_cluster(clusters, commits_dl, k_for_motifs=k,
                                             number_of_samples=self.num_motifs_to_sample,
                                             output_file=output_file_prefix.format(str(k)))
        else:
            motifs_by_cluster = mf.get_motifs_by_cluster(clusters, commits_dl, k_for_motifs=k,
                                         number_of_samples=self.num_motifs_to_sample,
                                         output_file=output_file_prefix.format(str(k)))
        print('{}-Length Motif by Cluster Outputted!'.format(k))
        return motifs_by_cluster
        #self.motifs_generated = True

    def get_multi_chain_percent(self, motifs_by_cluster, k):
        """f """
        multi_chain_perc = []
        for cluster in sorted(motifs_by_cluster.keys()):
            for motif in motifs_by_cluster[cluster]:
                if sum([motif.out_degree(node) in [0, 1] for node in motif]) == len(motif.nodes):
                    multi_chain_perc.append(self.num_motifs_to_sample-motifs_by_cluster[cluster][motif])
        multi_chain_perc_series = pd.Series(multi_chain_perc)
        multi_chain_perc_series.name = 'complexity_{}'.format(k)
        return multi_chain_perc_series

    def make_proj_stats_df(self, read_from_motif_files=False):
        # Load Data
        comm_auth_by_proj = pull_queries(COMM_AUTH_BY_PROJ.format(proj_ids=self.proj_ids_string)).set_index(
            'p_id')  # pd.read_csv('data/author_commits_by_proj_100.csv').set_index('p_id')
        pr_cr_by_proj = pull_queries(PR_CR_BY_PROJ.format(proj_ids=self.proj_ids_string)).set_index(
            'p_id')  # pd.read_csv('data/pr_cr_by_proj_100.csv').set_index('p_id')
        issues_by_proj = pull_queries(ISSUES_BY_PROJ.format(proj_ids=self.proj_ids_string)).set_index(
            'p_id')  # pd.read_csv('data/issues_by_proj_100.csv').set_index('p_id')
        owner_age_by_proj = pull_queries(OWNER_AGE_BY_PROJ.format(proj_ids=self.proj_ids_string)).set_index(
            'p_id')  # pd.read_csv('data/owner_age_by_proj_100.csv').set_index('p_id')
        time_betw_commits_by_proj = pull_queries(TBC_BY_PROJ.format(proj_ids=self.proj_ids_string)).set_index(
            'p_id')  # pd.read_csv('data/time_between_commits_100.csv').set_index('p_id')[['mean_tbc']]

        project = pd.concat([comm_auth_by_proj, pr_cr_by_proj, issues_by_proj, owner_age_by_proj, time_betw_commits_by_proj], axis=1)

        cluster_lookup = {}
        for cluster, value in self.clusters.items():
            for proj in value:
                cluster_lookup[proj] = cluster

        project['cluster'] = project.reset_index().p_id.apply(lambda x: cluster_lookup[x]).values

        multi_chain_percents = []
        for k in self.motif_lengths:
            motifs_by_cluster = self.generate_motifs(k, read_from_file=read_from_motif_files)
            multi_chain_percents.append(self.get_multi_chain_percent(motifs_by_cluster,k))

        complexity = pd.concat(multi_chain_percents, axis=1)
        project_stats = project.join(complexity, on='cluster', how='left')
        self.project_stats = project_stats

    def make_heatmap(self, output_path='./results/Report_VO.png'):
        # Normalize/Standardize
        # names = self.project_stats.drop('cluster', axis=1).columns
        # scaler = preprocessing.StandardScaler()
        # scaled_df = scaler.fit_transform(self.project_stats.drop('cluster', axis=1))
        # scaled_df = pd.DataFrame(scaled_df, columns=names, index=self.project_stats.index)
        # scaled_df = pd.merge(scaled_df, self.project_stats[['cluster']], left_index=True, right_index=True)

        sns.clustermap(self.project_stats.groupby('cluster').mean(),z_score=1,cmap='OrRd',col_cluster=False)#, cmap='RdYlGn')
        plt.savefig(output_path)



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
    hm = Heatmapper()
    hm.make_proj_stats_df(True)
    hm.make_heatmap()