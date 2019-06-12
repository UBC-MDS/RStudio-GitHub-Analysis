import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import pandas_gbq
import github_analysis.data_layer as dl
import github_analysis.motif_finder as mf
import github_analysis.freq_graph as fg
from google.oauth2.service_account import Credentials
from matplotlib.ticker import FuncFormatter
from decimal import Decimal
import panel as pn
from sklearn import preprocessing
import seaborn as sns; sns.set(color_codes=True)
from pdf2image import convert_from_path

from sklearn import preprocessing
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimread


def pull_queries(query_string, google_project_id='planar-elevator-238518', credentials_file='My Project 15448-4edea2614a66.json'):
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


class Dashboard:
    def __init__(self, gh_heatmap_file='./results/Report_gh_heatmap.png', lang_heatmap_file='./results/Report_lang_heatmap.png',
                 output_path='./results/Report_V1.pdf', embedding_path='results/embeddings.csv',data_path='/Users/richiezitomer/Documents/RStudio-Data-Repository/clean_data/commits.feather'):
        self.gh_heatmap_file = gh_heatmap_file
        self.lang_heatmap_file = lang_heatmap_file
        self.output_path = output_path

        self.emb = pd.read_csv(embedding_path)
        self.project_ids = self.emb.type.values
        self.proj_ids_string = ",".join(self.project_ids.astype(str))
        self.data_path = data_path
        self.commits_dl = dl.data_layer(data_path)

        self.images_made = True

    def make_overview_panel(self, output_path='./results/report_overview.png'):
        # Make Overview Author
        fig_commits,ax = plt.subplots()
        comm_auth_by_proj = pull_queries(COMM_AUTH_BY_PROJ.format(proj_ids=self.proj_ids_string)).set_index(
            'p_id')
        sns.kdeplot(comm_auth_by_proj.commits, shade=True,ax=ax)
        #ax.set_xlim([0, 20000])
        sns.despine()
        ax.legend('')
        ax.set_title('Commits Density Plot')
        ax.set_ylabel('Density')
        ax.set_xlabel('# of Commits per Project')
        formatter = FuncFormatter(shorten_decimal)
        ax.yaxis.set_major_formatter(formatter)
        plt.yticks(rotation='vertical')

        # Make Commits Histograms
        fig_authors,ax = plt.subplots()
        sns.kdeplot(comm_auth_by_proj.authors, shade=True,ax=ax)
        # ax.set_xlim([0, 20000])
        sns.despine()
        ax.legend('')
        ax.set_title('Author Density Plot')
        ax.set_ylabel('Density')
        ax.set_xlabel('# of Authors per Project')
        plt.tight_layout()

        # Make Overview Motif Graph
        overall_motifs = mf.get_motifs(self.project_ids,k_for_motifs=5,number_of_samples=100,data_layer=self.commits_dl)
        fig_motifs =fg.visualize_motif_samples_bar_graph(overall_motifs)
        overview = pn.Column(
            pn.Row(fig_motifs,sizing_mode='stretch_width',margin=(0,100)),
            pn.Row(
            pn.Column(fig_commits),
            pn.Column(fig_authors)
            ))
        overview.save(output_path)

    def make_heatmaps(self, output_path='./results/report_heatmap.png'):
        # Make heatmaps for cluster comparison

        # pn.Column(
        #     pn.Row(fig_motifs, sizing_mode='stretch_width', margin=(0, 100)),
        #     pn.Row(
        #         pn.Column(fig_commits),
        #         pn.Column(fig_authors)
        #     ))
        gh_heatmap = self.make_gh_heatmap()
        gh_heatmap.savefig('./results/report_gh_heatmap.png')
        lang_heatmap = self.make_lang_heatmap()
        lang_heatmap.savefig('./results/report_lang_heatmap.png')
        # hm = Heatmapper()
        # hm.make_proj_stats_df(True)
        # hm.make_heatmap()
        # # txt = 'Cluster Comparison'
        # # plt.text(0.025, 0.99, txt, transform=fig.transFigure, size=24)
        # plt.tight_layout()
        #
        # hm.make_heatmap()
        # plt.tight_layout()
        heatmap = pn.Row(
            pn.Column(gh_heatmap),
            pn.Column(lang_heatmap)
        )
        heatmap.save(output_path)

    def make_lang_radial_plots(self,output_path='./results/report_lang_radial_plot.png',mode='panel'):
        # Set up language plotter
        lang_plotter = LanguagePlotter()
        lang_plotter.get_clusters()
        lang_plotter.set_languages('project_languages.csv')
        _ = lang_plotter.get_top_languages(5, 15)

        fig = plt.figure(figsize=(20, 20))

        rows = 5
        columns = 4

        gs = fig.add_gridspec(rows, columns, hspace=0.5)

        for i in range(rows *columns):
            ax = fig.add_subplot(gs[i], polar=True)
            lang_plotter.radial_plotter(i, ax=ax, mode=mode)
        fig.savefig(output_path)


    def make_lang_heatmap(self):
        lang_plotter = LanguagePlotter()
        lang_plotter.get_clusters()
        lang_plotter.set_languages('project_languages.csv')
        lang_plotter.get_top_languages(5, 15)
        return lang_plotter.heatmap_cluster()

    def make_gh_heatmap(self):
        hm = Heatmapper()
        hm.make_proj_stats_df(True)
        return hm.make_heatmap() #'./results/Report_lang_heatmap.png')

    def make_cluster_motifs():
        pages = convert_from_path('.results/clustering_output.pdf', 500)
        for n, page in enumerate(pages):
            page.save('cluster_{}_motif.png'.format(n))


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
                motifs_by_cluster = mf.get_motifs_by_cluster(self.clusters, self.commits_dl, k_for_motifs=k,
                                             number_of_samples=self.num_motifs_to_sample,
                                             output_file=output_file_prefix.format(str(k)))
        else:
            motifs_by_cluster = mf.get_motifs_by_cluster(self.clusters, self.commits_dl, k_for_motifs=k,
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
        multi_chain_perc_series.name = 'mcp_{}'.format(k)
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

    def make_heatmap(self, palette='OrRd', palette_levels=10):#,output_path='./results/Report_gh_heatmap.png',ax=None):
        #fig = plt.figure()
        cmp = sns.clustermap(self.project_stats.groupby('cluster').mean().T,z_score=0,
                             cmap=sns.color_palette(palette, palette_levels),row_cluster=False)#cmap='OrRd',col_cluster=False,figsize=(6,4))#, cmap='RdYlGn')
        # fig.savefig(output_path)
        plt.suptitle('GitHub Heatmap by Cluster')
        return cmp.fig


class LanguagePlotter:
    def __init__(self, cluster_path='./results/clusters.pickle'):
        self.k_means = None
        self.project_clusters = None
        self.project_langauges = None
        self.cluster_path = cluster_path

    def get_clusters(self):
        pickle_in = open(self.cluster_path, "rb")
        clusters = pickle.load(pickle_in)
        proj_id = []
        cluster = []
        for k in clusters:
            for pid in clusters[k]:
                proj_id.append(pid)
                cluster.append(k)
        self.project_clusters = pd.DataFrame({'project_id': proj_id,
                                              'cluster': cluster})

    def set_languages(self, languages_path):
        languages = pd.read_csv(languages_path)
        project_languages = pd.merge(languages, self.project_clusters, on='project_id')
        self.project_languages = project_languages

    def get_top_languages(self, min_proj, min_pct):
        total_bytes = self.project_languages.groupby(['project_id']).agg({'bytes': 'sum'})
        total_bytes.reset_index(inplace=True)
        total_bytes.columns = pd.Index(['project_id', 'total_bytes'])

        self.project_languages = pd.merge(self.project_languages, total_bytes)
        self.project_languages['lang_pct'] = self.project_languages['bytes'] / self.project_languages[
            'total_bytes'] * 100

        cluster_language = pd.DataFrame(self.project_languages.groupby(
            ['cluster', 'language']).agg(
            {'lang_pct': ['mean', 'std', 'median', 'count']})).reset_index()

        cluster_language.columns = ['cluster', 'language',
                                    'mean_lang_pct', 'std_lang_pct',
                                    'median_lang_pct', 'n']

        self.raw_cluster_data = cluster_language.copy()
        cluster_data = cluster_language.query(f'mean_lang_pct > {min_pct} and n > {min_proj}')[
            ['cluster', 'language', 'mean_lang_pct']]
        cluster_data = cluster_data.pivot(index='cluster', columns='language', values='mean_lang_pct')
        lang_of_interest = cluster_data.apply(get_percentage_missing).index[
            cluster_data.apply(get_percentage_missing) < 0.25]
        cluster_data = cluster_data[lang_of_interest.values].fillna(0)

        self.cluster_data = cluster_data
        return cluster_data

    def heatmap_cluster(self, palette='OrRd', palette_levels=5):
        # scaler = preprocessing.StandardScaler()
        # langs = self.cluster_data.columns
        # cluster_data = scaler.fit_transform(self.cluster_data)
        # cluster_data = pd.DataFrame(cluster_data, columns=langs)
        #
        cmp = sns.clustermap(self.cluster_data.T, z_score=0,cmap=sns.color_palette(palette, palette_levels))
        plt.suptitle('Language Heatmap by Cluster')
        return cmp.fig

    def radial_plotter(self, cluster_id, **kwargs):
        return plot_radial(self.raw_cluster_data, cluster_id, **kwargs)


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
    # hm = Heatmapper()
    # hm.make_proj_stats_df(True)
    # hm.make_heatmap()
    d = Dashboard()
    # d.make_overview_panel()
    d.make_heatmaps()
    # d.make_lang_radial_plots()
   # d.make_lang_radial_plots(output_path='./results/report_lang_radial_plot_indiv.png',mode='individual')


