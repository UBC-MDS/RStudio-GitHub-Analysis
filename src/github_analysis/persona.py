from collections import defaultdict
import numpy as np

def get_persona_projects(clusters, dl, n, seed):
    if seed:
        np.random.seed(seed)

    personas = defaultdict(list)
    for cluster in clusters:
        project_ids_in_cluster = clusters[cluster]
        if len(project_ids_in_cluster) <= n:
            for pid in project_ids_in_cluster:
                personas[cluster].append(dl.getProjectNameById(pid))
            continue


        # Random choice could be improved to take projects closer to the mean of the cluster
        random_personas = np.random.choice(project_ids_in_cluster, n, replace=False)

        for rp in random_personas:
            personas[cluster].append(dl.getProjectNameById(rp))

    print(personas)
    return personas
