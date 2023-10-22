BZR_best_params = {
                        "gamma":0.05 ,
                        "max_depth":3,
                        "subsample":0.65,
                        "reg_lambda":0.3,
                        "reg_alpha":0.65,
                        "n_estimators":250,
                        "learning_rate": 0.01
                    }

COX2_best_params = {
                        "gamma":0.2,
                        "max_depth":5,
                        "subsample":0.8,
                        "reg_lambda": 0.6,
                        "reg_alpha":0.4,
                        "n_estimators":200,
                        "learning_rate": 0.0001
                    }


DD_best_params =  {
                        "gamma":0.2,
                        "max_depth":7,
                        "subsample":0.55,
                        "reg_lambda":0.85,
                        "reg_alpha":0.1,
                        "n_estimators":200,
                        "learning_rate": 0.01
                    }


ENZYMES_best_params =  {
                        "gamma":0.4,
                        "max_depth":5,
                        "subsample":0.85,
                        "reg_lambda":0.8,
                        "reg_alpha":0.7,
                        "n_estimators":300,
                        "learning_rate":0.01
                    }

COLLAB_best_params = {
                        "gamma":0.1,
                        "max_depth":6,
                        "subsample":0.85,
                        "reg_lambda":0.35,
                        "reg_alpha":0.6,
                        "n_estimators":250,
                        "learning_rate": 0.1
                    }
IMDB_MULTI_best_params = {
                        "gamma":0.25,
                        "max_depth":3,
                        "subsample":0.6,
                        "reg_lambda":0.4,
                        "reg_alpha":0.05,
                        "n_estimators":150,
                        "learning_rate": 0.01
                    }
#
NCI1_best_params = {
                        "gamma":0.35,
                        "max_depth":7,
                        "subsample":0.75,
                        "reg_lambda":0.1,
                        "reg_alpha":0.15,
                        "n_estimators":250,
                        "learning_rate":0.01
                    }

PROTEINS_best_params =  {
                        "gamma":0.05,
                        "max_depth":5,
                        "subsample":0.55,
                        "reg_lambda":0.65,
                        "reg_alpha":0.5,
                        "n_estimators":150,
                        "learning_rate":0.01
                    }

PTC_MR_best_params =  {
                        "gamma":0.3,
                        "max_depth":5,
                        "subsample":0.9,
                        "reg_lambda":0.5,
                        "reg_alpha":0.3,
                        "n_estimators":200,
                        "learning_rate": 0.001
                    }
NCI109_best_params =  {
                        "gamma":0.25,
                        "max_depth":8,
                        "subsample":0.65,
                        "reg_lambda":0.95,
                        "reg_alpha":0.75,
                        "n_estimators":100,
                        "learning_rate": 0.01
                    }

IMDB_BINARY_best_params = {
                        "gamma":0.5,
                        "max_depth":4,
                        "subsample":0.75,
                        "reg_lambda":0.25,
                        "reg_alpha":1,
                        "n_estimators":200,
                        "learning_rate": 0.01
                    }

enron_best_params = {
                        "gamma":0.3,
                        "max_depth":5,
                        "subsample":0.85,
                        "reg_lambda":0.4,
                        "reg_alpha":0.6,
                        "n_estimators":300,
                        "learning_rate": 0.0001
                    }

reality_mining_best_params = {
                        "gamma":0.1,
                        "max_depth":4,
                        "subsample":0.65,
                        "reg_lambda":0.25,
                        "reg_alpha":0.5,
                        "n_estimators":200,
                        "learning_rate": 0.001
                    }

sec_repo_best_params = {
                        "gamma":0.05,
                        "max_depth":4,
                        "subsample":0.75,
                        "reg_lambda":0.15,
                        "reg_alpha":1,
                        "n_estimators":100,
                        "learning_rate": 0.001
                    }

twitter_security_best_params = {
                        "gamma":0.25,
                        "max_depth":9,
                        "subsample":0.55,
                        "reg_lambda":0.05,
                        "reg_alpha":0.9,
                        "n_estimators":200,
                        "learning_rate": 0.01
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
    elif dataset_name=="COLLAB":
        return COLLAB_best_params
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