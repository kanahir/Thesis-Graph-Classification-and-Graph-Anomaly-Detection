
BZR_best_params = {
            "NodeDropping": 90,
            "EdgePerturbation": 30,
            "AttributeMasking": 55,
            "Subgraph": 100,
            "Louvain": 85,
            "NodeEmbedding": 40,
            "ID": 30,
                    }

DD_best_params = {
            "NodeDropping": 55,
            "EdgePerturbation": 85,
            "AttributeMasking": 30,
            "Subgraph": 65,
            "Louvain": 50,
            "NodeEmbedding": 80,
            "ID": 55,
                    }


NCI1_best_params = {
            "NodeDropping": 70,
            "EdgePerturbation": 85,
            "AttributeMasking": 65,
            "Subgraph": 65,
            "Louvain": 95,
            "NodeEmbedding": 35,
            "ID": 90,
                    }

COX2_best_params = {
            "NodeDropping": 80,
            "EdgePerturbation": 25,
            "AttributeMasking": 60,
            "Subgraph": 25,
            "Louvain": 10,
            "NodeEmbedding": 50,
            "ID": 65,
                    }

ENZYMES_best_params = {
            "NodeDropping": 55,
            "EdgePerturbation": 80,
            "AttributeMasking": 30,
            "Subgraph": 60,
            "Louvain": 30,
            "NodeEmbedding": 25,
            "ID": 25,
                    }

PROTEINS_best_params = {
            "NodeDropping": 25,
            "EdgePerturbation": 50,
            "AttributeMasking": 15,
            "Subgraph": 50,
            "Louvain": 70,
            "NodeEmbedding": 60,
            "ID": 55,
                    }

PTC_MR_best_params = {
            "NodeDropping": 30,
            "EdgePerturbation": 20,
            "AttributeMasking": 60,
            "Subgraph": 75,
            "Louvain": 80,
            "NodeEmbedding": 70,
            "ID": 50
}

IMDB_BINARY_best_params = {
            "NodeDropping": 30,
            "EdgePerturbation": 60,
            "AttributeMasking": 50,
            "Subgraph": 75,
            "Louvain": 35,
            "NodeEmbedding": 80,
            "ID": 75
}

IMDB_MULTI_best_params = {
            "NodeDropping": 65,
            "EdgePerturbation": 45,
            "AttributeMasking": 90,
            "Subgraph": 40,
            "Louvain": 60,
            "NodeEmbedding": 70,
            "ID": 40
}
NCI109_best_params = {
            "NodeDropping": 95,
            "EdgePerturbation": 60,
            "AttributeMasking": 80,
            "Subgraph": 50,
            "Louvain": 75,
            "NodeEmbedding": 50,
            "ID": 80
}
sec_repo_best_params = {
            "NodeDropping": 50,
            "EdgePerturbation": 55,
            "AttributeMasking": 55,
            "Subgraph": 95,
            "Louvain": 10,
            "NodeEmbedding": 40,
            "ID": 65
}
enron_best_params = {
            "NodeDropping": 20,
            "EdgePerturbation": 45,
            "AttributeMasking": 30,
            "Subgraph": 95,
            "Louvain": 50,
            "NodeEmbedding": 40,
            "ID": 85
}
reality_mining_best_params  = twitter_security_best_params  = {
            "NodeDropping": 60,
            "EdgePerturbation": 40,
            "AttributeMasking": 70,
            "Subgraph": 80,
            "Louvain": 60,
            "NodeEmbedding": 45,
            "ID": 35
}

def get_best_params(dataset_name):
    if dataset_name=="BZR":
        return BZR_best_params
    elif dataset_name=="COX2":
        return COX2_best_params
    elif dataset_name=="DD":
        return DD_best_params
    elif dataset_name=="ENZYMES":
        return ENZYMES_best_params
    elif dataset_name=="IMDB-MULTI":
        return IMDB_MULTI_best_params
    elif dataset_name=="NCI1":
        return NCI1_best_params
    elif dataset_name=="PROTEINS_full":
        return PROTEINS_best_params
    elif dataset_name=="PTC_MR":
        return PTC_MR_best_params
    elif dataset_name=="NCI109":
        return NCI109_best_params
    elif dataset_name=="IMDB-BINARY":
        return IMDB_BINARY_best_params
    elif dataset_name=="enron":
        return enron_best_params
    elif dataset_name=="reality_mining":
        return reality_mining_best_params
    elif dataset_name=="sec_repo":
        return sec_repo_best_params
    elif dataset_name=="twitter_security":
        return twitter_security_best_params