import pandas as pd
import pickle
import matplotlib.pyplot as plt
import argparse
import numpy as np
import seaborn as sns
import os
import pandas_gbq
import data_layer as dl
import motif_finder as mf
import freq_graph as fg
from google.oauth2.service_account import Credentials
from matplotlib.ticker import FuncFormatter
from decimal import Decimal
import networkx as nx
import panel as pn
from random import sample
from sklearn import preprocessing
import seaborn as sns; sns.set(color_codes=True)
from pdf2image import convert_from_path
import joypy
from nxutils import git_graph
import scipy.stats as st

from sklearn import preprocessing
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimread


def pull_queries(query_string, google_project_id='planar-elevator-238518', credentials_file='credentials_file.json'):
    credentials = Credentials.from_service_account_file(credentials_file)
    return pandas_gbq.read_gbq(query_string,project_id=google_project_id,credentials=credentials)


def shorten_decimal(x, pos):
    'The two args are the value and tick position'
    return '%.E' % Decimal(x)


def get_percentage_missing(series):
    """ Calculates percentage of NaN values in DataFrame
    :param series: Pandas DataFrame object
    :return: float
    """
    num = series.isnull().sum()
    den = len(series)
    return round(num/den, 2)


def plot_radial(cluster_df, cluster_id, ax, mode, n=50):
    if mode == 'panel':
        langs_of_interest = cluster_df.query(f'n > {n}').groupby(['language'])['mean_lang_pct'].mean().sort_values(
            ascending=False).iloc[0:8].index
    elif mode == 'individual':
        langs_of_interest = cluster_df.query(f'cluster == {int(cluster_id)} and n > {n}').language

    cluster = cluster_df[cluster_df.language.isin(langs_of_interest)]
    cluster = cluster.query(f'cluster == {int(cluster_id)}')

    angles = [n / float(cluster.shape[0]) * 2 * np.pi for n in range(cluster.shape[0])]
    angles += angles[:1]

    values = cluster.mean_lang_pct.values
    values = np.append(values, values[:1])

    ci_up = cluster.mean_lang_pct.values + (cluster.std_lang_pct / np.sqrt(cluster.n)) * 1.96
    ci_up = np.append(ci_up, ci_up[:1])

    ci_down = cluster.mean_lang_pct.values - (cluster.std_lang_pct / np.sqrt(cluster.n)) * 1.96
    ci_down = np.append(ci_down, ci_down[:1])

    angles = [n / float(cluster.shape[0]) * 2 * np.pi for n in range(cluster.shape[0])]
    angles += angles[:1]

    # Add labels to the languages
    languages = cluster.language.str.capitalize()
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(languages, fontdict=None, minor=False)

    # Plots the estimate
    ax.scatter(angles, values, alpha=0.5)

    # Plots the inside area
    ax.fill(angles, values, 'lightsalmon', alpha=0.75)
    ax.set_rgrids((0, 20, 40, 60, 80), ('', '', '', '', ''))

    plt.title(f'Cluster {cluster_id}\n')
    return ax

def complexity_tag(x):
    if x>.75: #60%
        return 'high_complexity'
    elif x< .40:
        return 'low_complexity'
    else:
        return None

def calc_conf(a):
    """Adapted from answers to this question: https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data"""
    interval = st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))
    return interval[1]-interval[0]

class Report:
    """Class to generate images used in report and presentation."""
    def __init__(self, data_path='/Users/richiezitomer/Documents/RStudio-Data-Repository/clean_data/commits_by_org.feather',
                 embedding_path='results/embeddings.csv', num_motifs_to_sample=1000, motif_lengths=[5,10,25,50,100]):
        self.emb = pd.read_csv(embedding_path)
        self.project_ids = self.emb.type.values
        self.proj_ids_string = ",".join(self.project_ids.astype(str))
        self.data_path = data_path
        self.commits_dl = dl.data_layer(data_path)
        self.num_motifs_to_sample = num_motifs_to_sample
        self.motif_lengths = motif_lengths

        self.project_stats_created = False

    def make_proj_stats_df(self):
        """Method to make dataframe with stats by project."""
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
        project['p_id'] = project.index
        multi_chain_percents = []
        for k in self.motif_lengths:
            multi_chain_perc_series = project.p_id.apply(lambda x: self.get_multi_chain_percent_by_proj(k,x))
            multi_chain_perc_series.name = 'mcp_{}'.format(k)
            multi_chain_percents.append(multi_chain_perc_series)
        complexity = pd.concat(multi_chain_percents, axis=1)
        project_stats = project.join(complexity, how='left')

        self.project_stats = project_stats
        self.project_stats_created = True

    def get_multi_chain_percent_by_proj(self, k,proj_id):
        """Method that gets multi-chain percentage of each project."""
        projects_cluster = self.commits_dl.getCommitsByProjectId(proj_id)
        G = git_graph(projects_cluster)
        roots = [n for n, d in G.in_degree() if d == 0]

        mcs = 0
        scs = 0
        if len(roots) > 10:
            roots = sample(roots, 10)
        for root in roots:
            edges = nx.bfs_edges(G,root)  # https://networkx.github.io/documentation/networkx-2.2/reference/algorithms/generated/networkx.algorithms.traversal.breadth_first_search.bfs_edges.html#networkx.algorithms.traversal.breadth_first_search.bfs_edges
            nodes = [root] + [v for u, v in edges]
            #    print(len(nodes))

            for i in range(0, min(len(nodes),200), k):
                current_root = nodes[i]
                current_edges = nx.bfs_edges(G, current_root)  # https://networkx.github.io/documentation/networkx-2.2/reference/algorithms/generated/networkx.algorithms.traversal.breadth_first_search.bfs_edges.html#networkx.algorithms.traversal.breadth_first_search.bfs_edges
                current_nodes = [current_root] + [v for u, v in current_edges]
                if len(current_nodes) < k:
                    continue
                subgraph = G.subgraph(current_nodes[:k])
                if sum([subgraph.out_degree(node) in [0, 1] for node in subgraph]) != k:
                    mcs += 1
                else:
                    scs += 1
        if scs+mcs == 0:
            return None
        else:
            return mcs/(scs+mcs)

    # def make_lang_radial_plots(self,output_path='./results/report_lang_radial_plot.png',mode='panel'):
    #     # Set up language plotter
    #     lang_plotter = LanguagePlotter()
    #     lang_plotter.get_clusters()
    #     lang_plotter.set_languages('project_languages.csv')
    #     _ = lang_plotter.get_top_languages(5, 15)
    #
    #     fig = plt.figure(figsize=(20, 20))
    #
    #     rows = 5
    #     columns = 4
    #
    #     gs = fig.add_gridspec(rows, columns, hspace=0.5)
    #
    #     for i in range(rows *columns):
    #         ax = fig.add_subplot(gs[i], polar=True)
    #         lang_plotter.radial_plotter(i, ax=ax, mode=mode)
    #     fig.savefig(output_path)

    def get_most_common_motifs(self, motif_length=5):
        """Method that gets 8 or 9 most common motifs for a given project or group of projects."""
        motifs = mf.get_motifs(self.project_ids, motif_length, self.num_motifs_to_sample, self.commits_dl)

        if motif_length == 5:
            fig, axs = plt.subplots(3, 3)
        else:
            fig, axs = plt.subplots(4, 2)

        fig.set_size_inches(18.5, 10.5)
        for n, key in enumerate(sorted(motifs, key=motifs.get, reverse=True)):
            if motif_length == 5:
                if n >= 9:
                    break
                nx.draw_kamada_kawai(key, node_size=300, width=1.5, arrowsize=50, ax=axs.flatten()[n])
                axs.flatten()[n].set_title(
                    '{}. {}% (n={})'.format(str(n + 1), str(round(100*(motifs[key] / self.num_motifs_to_sample))), str(motifs[key])),
                    fontsize=20)
            else:
                if n >= 8:
                    break
                if n == 0:
                    nx.draw_kamada_kawai(key, node_size=100, width=1, ax=axs.flatten()[n])
                    axs.flatten()[n].set_title('{}. {}% (n={})'.format(str(n + 1), str(round(100 * (motifs[key] / self.num_motifs_to_sample))),
                                            str(motifs[key])),fontsize = 20)
                else:
                    nx.draw_spring(key, node_size=100, width=.8, arrowsize=20, ax=axs.flatten()[n])
                    axs.flatten()[n].set_title('{}. {}% (n={})'.format(str(n + 1), str(round(100 * (motifs[key] / self.num_motifs_to_sample))),
                                            str(motifs[key])),fontsize = 20)

        fig.suptitle('Most Common Motifs Length {} Occurrence Rate and Count'.format(motif_length), fontsize=25)
        fig.savefig('results/motif_{}_visual.png'.format(motif_length))
        return fig

    def get_motif_example(self, motif_length=25):
        """Method that gets an example motif of motif_length."""
        motifs = mf.get_motifs(self.project_ids, motif_length, self.num_motifs_to_sample, self.commits_dl)
        second_most_common_motif = sorted(motifs, key=motifs.get, reverse=True)[1]

        fig, ax = plt.subplots()
        nx.draw_spring(second_most_common_motif, node_size=100, ax=ax)
        fig.suptitle('Common Git Motif \n Length {}'.format(motif_length), fontsize=20)

        fig.savefig('results/motif_example.png')
        return fig

    def get_mcp_hist(self):
        """Method that makes a histogram of different motif lengths by project."""
        if not self.project_stats_created:
            self.make_proj_stats_df()
        df = self.project_stats[['mcp_5', 'mcp_10', 'mcp_25', 'mcp_50', 'mcp_100']]
        df.columns = ['Length 5', 'Length 10', 'Length 25', 'Length 50', 'Length 100']

        fig, axes = joypy.joyplot(df,
                                  title='Distribution of Commit Chains With at Least \n One Branch or Merge, by Chain Length')
        axes[-1].set_xlabel('Ratio of Chains With at Least One Branch or Merge')
        fig.tight_layout()
        fig.savefig('results/mcp_histograms.png')
        return fig

    def get_gh_feature_comparison(self):
        """Method that gets relative GH features of high- and low-complexity projects."""
        if not self.project_stats_created:
            self.make_proj_stats_df()
        self.project_stats['complexity'] = self.project_stats.mcp_25.apply(complexity_tag)

        high_low = self.project_stats.groupby('complexity')[
            ['commits', 'authors', 'issues', 'prs', 'code_reviews', 'owner_age', 'mean_tbc']].mean().T

        issues_ci_high = calc_conf(self.project_stats[self.project_stats.complexity == 'high_complexity'].issues)
        issues_ci_low = calc_conf(self.project_stats[self.project_stats.complexity == 'low_complexity'].issues)

        prs_ci_high = calc_conf(self.project_stats[self.project_stats.complexity == 'high_complexity'].prs)
        prs_ci_low = calc_conf(self.project_stats[self.project_stats.complexity == 'low_complexity'].prs)

        cr_ci_high = calc_conf(self.project_stats[self.project_stats.complexity == 'high_complexity'].code_reviews)
        cr_ci_low = calc_conf(self.project_stats[self.project_stats.complexity == 'low_complexity'].code_reviews)

        yerr = np.array([[issues_ci_high, issues_ci_low], [prs_ci_high, prs_ci_low], [cr_ci_high, cr_ci_low]])

        high_low_errors = pd.DataFrame(yerr, index=['issues', 'prs', 'code_reviews'],
                                       columns=['high_complexity', 'low_complexity'])

        fig, ax = plt.subplots()
        high_low.drop(['authors', 'commits', 'owner_age', 'mean_tbc']).plot(kind='bar', ax=ax, yerr=high_low_errors)
        ax.set_xlabel('GitHub Features')
        ax.set_ylabel('Average Count')
        ax.set_title('Average Issues, PRs, and Code Reviews \n by Project for High- and Low-Complexity Git Graphs')
        ax.set_xticklabels(['Issues', 'Pull \n Requests', 'Code \n Reviews'])
        plt.xticks(rotation=360)

        fig.savefig('results/GH_features_by_complexity')

        return fig


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
    and date_diff(DATE(p.created_at), DATE(u.created_at),DAY)>=0
    and EXTRACT(YEAR FROM p.created_at)>=2011
    and EXTRACT(YEAR FROM p.created_at)<=2016
    and EXTRACT(YEAR FROM u.created_at)>=2011
    and EXTRACT(YEAR FROM u.created_at)<=2016"""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp",      "--data_path",      help="The path to the commits.feather file. e.g. /home/user/RStudio-Data-Repository/clean_data/commits_by_org.feather", default="/Users/richiezitomer/Documents/RStudio-Data-Repository/clean_data/commits_by_org.feather")
    parser.add_argument("-ep",      "--embedding_path", help="The path to the embeddings file. e.g. results/embeddings.csv", default="results/embeddings.csv")
    args = parser.parse_args()

    r = Report(data_path=args.data_path, embedding_path=args.embedding_path)
    r.get_most_common_motifs()
    r.get_most_common_motifs(motif_length=25)
    r.get_motif_example()
    r.get_mcp_hist()
    r.get_gh_feature_comparison()
    print('Report Images Run!')
