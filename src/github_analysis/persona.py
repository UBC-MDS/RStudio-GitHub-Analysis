import csv
import logging
import webbrowser
from collections import defaultdict

import numpy as np

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",
                    filename="log.log", level=logging.INFO)

def convert_api_url_to_project_url(url):
    # https://api.github.com/repos/agentk/Catstagram,
    url = str(url)
    url = url.replace("/api.", "/")
    url = url.replace("/repos/", "/")
    return url

class Personas:
    def __init__(self, clusters, dl, n, seed, output_path):
        self.clusters = clusters
        self.dl = dl
        self.n = n
        self.seed = seed
        self.output_path = output_path

        self.personas = self.get_persona_projects(self.clusters, self.dl, self.n, self.seed, self.output_path)

    def get_persona_projects(self, clusters, dl, n, seed, output_path):
        if seed:
            np.random.seed(seed)

        personas = defaultdict(list)

        for cluster in clusters:
            project_ids_in_cluster = clusters[cluster]
            if len(project_ids_in_cluster) <= n:
                for pid in project_ids_in_cluster:
                    project_url = dl.getProjectURLById(pid)
                    personas[cluster].append(
                        convert_api_url_to_project_url(project_url))
                continue

            # Random choice could be improved to take projects closer to the mean of the cluster
            random_personas = np.random.choice(
                project_ids_in_cluster, n, replace=False)

            for rp in random_personas:
                project_url = dl.getProjectURLById(rp)
                personas[cluster].append(
                    convert_api_url_to_project_url(project_url))

        self.personas = personas
        logging.info(personas)

        with open(output_path, 'w') as file:
            w = csv.DictWriter(file, personas.keys())
            w.writeheader()
            w.writerow(personas)

        return personas

    def open_personas_in_browser(self, cluster_key):
        websites = self.personas[cluster_key]
        for website in websites:
            webbrowser.open(website)
