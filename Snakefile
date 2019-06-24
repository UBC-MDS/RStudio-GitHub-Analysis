rule run_analysis:
    input:
        data_path = "/home/user/RStudio-Data-Repository/clean_data/commits_by_org.feather"
    output:
        results_path = directory("results")
    params:
        python_hash_seed = 0,
        n_workers = 8,
        n_projects = 1000,
        min_commits = None,
        min_count = 5,
        n_personas = 5,
        n_neurons = 128,
        n_iter = 10,
        random_state = 1
    shell:
        "PYTHONHASHSEED={params.python_hash_seed} python src/github_analysis/main.py -dp {input.data_path} -rp {output.results_path} -nw {params.n_workers} -np {params.n_projects} -mc {params.min_commits} -mcount {params.min_count} -nps {params.n_personas} -nn {params.n_neurons} -ni {params.n_iter} -rs {params.random_state}"

# Commented out because repo is currently over bandwidth: https://help.github.com/en/articles/about-storage-and-bandwidth-usage
#rule clone_data_repo:
#    shell: "git clone https://github.com/UBC-MDS/RStudio-Data-Repository.git"
