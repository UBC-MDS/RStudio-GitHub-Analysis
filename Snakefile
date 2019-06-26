configfile: "config.json"

rule run_analysis:
    input:
        data_path = "/Users/richiezitomer/Documents/RStudio-Data-Repository/clean_data/commits_by_org.feather"
    output:
        results_path = directory("results/")
    params:
        python_hash_seed = config["python_hash_seed"],
        n_workers = config["n_workers"],
        n_projects = config["n_projects"],
        min_commits = config["min_commits"],
        min_count = config["min_count"],
        n_personas = config["n_personas"],
        n_neurons = config["n_neurons"],
        n_iter = config["n_iter"],
        random_state = config["random_state"]
    shell:
        "PYTHONHASHSEED={params.python_hash_seed} python src/github_analysis/main.py -dp {input.data_path} -rp {output.results_path} -nw {params.n_workers} -np {params.n_projects} -mc {params.min_commits} -mcount {params.min_count} -nps {params.n_personas} -nn {params.n_neurons} -ni {params.n_iter} -rs {params.random_state}"

rule generate_images:
    input:
        data_path="/Users/richiezitomer/Documents/RStudio-Data-Repository/clean_data/commits_by_org.feather",
        embedding_path="results/embeddings.csv"
    shell:
        "python src/github_analysis/make_report.py -dp {input.data_path} -ep {input.embedding_path}"


# Commented out because repo is currently over bandwidth: https://help.github.com/en/articles/about-storage-and-bandwidth-usage
#rule clone_data_repo:
#    shell: "git clone https://github.com/UBC-MDS/RStudio-Data-Repository.git"
