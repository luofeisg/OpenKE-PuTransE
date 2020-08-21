import sys
from pathlib import Path
import torch

openke_path = Path.cwd().parents[0]
print(openke_path)
sys.path.append(str(openke_path))

# OpenKE modules
from openke.module.model import TransH
from openke.data import TrainDataLoader, TestDataLoader
from openke.config import Trainer, Tester, Validator
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from copy import deepcopy


def get_hyper_param_permutations(transe_hyper_param_dict):
    permutated_hyperparam_dict = [{"norm": norm, "margin": margin, "learning_rate": lr, "dimension": dim}
                                  for norm in transe_hyper_param_dict["norm"]
                                  for margin in transe_hyper_param_dict["margin"]
                                  for lr in transe_hyper_param_dict["learning_rate"]
                                  for dim in transe_hyper_param_dict["dimension"]
                                  ]

    return permutated_hyperparam_dict


def train_Model(model, hyper_param_dict, dataset_name, experiment_index, dataset_path, valid_steps, early_stop_patience,
                 max_epochs):
    init_random_seed = 4
    print("Initial random seed is:", init_random_seed)

    # get_hyper_params
    norm = hyper_param_dict["norm"]
    margin = hyper_param_dict["margin"]
    lr = hyper_param_dict["learning_rate"]
    dim = hyper_param_dict["dimension"]

    print("Train with:")
    print("norm: {}".format(norm))
    print("margin: {}".format(margin))
    print("learning_rate: {}".format(lr))
    print("dimension: {}".format(dim))

    # (1) Initialize TrainDataLoader for sampling of examples
    train_dataloader = TrainDataLoader(
        in_path=dataset_path,
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=0,
        neg_ent=1,
        neg_rel=0,
        random_seed=init_random_seed)

    # (2) Create embedding model
    transe = model(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=dim,
        p_norm=norm,
        norm_flag=True)

    # define the loss function
    model = NegativeSampling(
        model=transe,
        loss=MarginLoss(margin=margin),
        batch_size=train_dataloader.get_batch_size()
    )

    # -- Validator
    valid_dl = TestDataLoader(train_dataloader.in_path, "link", mode='valid')
    validator = Validator(model=transe, data_loader=valid_dl)

    # -- Validation params
    valid_steps = valid_steps
    early_stopping_patience = early_stop_patience
    bad_counts = 0
    best_hit10 = 0
    bad_count_limit_reached = False
    best_model = None

    # trainer to train the model with trainer.run(
    trainer = Trainer(model=model, data_loader=train_dataloader, alpha=lr, train_times=valid_steps,
                      use_gpu=torch.cuda.is_available())

    trained_epochs = 0

    # while(early_stopping_patience !=0):
    while not bad_count_limit_reached and trained_epochs < max_epochs:
        trainer.run()
        trained_epochs += valid_steps

        # Validation
        # TEST
        hit10 = validator.valid()
        # hit10 = 2 * trained_epochs * experiment_index

        print("hits@10 is: {}.".format(hit10))
        if hit10 > best_hit10:
            best_hit10 = hit10
            print("Best model | hit@10 of valid set is %f" % best_hit10)
            print('Save model at epoch %d.' % trained_epochs)
            best_model = deepcopy(transe)
            transe.save_checkpoint(
                '../checkpoint/transe_{}_exp{}.ckpt'.format(dataset_name, experiment_index))
            bad_counts = 0
        else:
            bad_counts += 1
            print(
                "Hit@10 of valid set is %f | bad count is %d"
                % (hit10, bad_counts)
            )

        if bad_counts == early_stopping_patience:
            bad_count_limit_reached = True
            print("----> Early stopping at epoch {}".format(trained_epochs))
            break

    return best_model


def test_model(model, dataset_path):
    # -- Validator
    test_dataloader = TestDataLoader(dataset_path, "link", mode='test')
    tester = Tester(model=model, data_loader=test_dataloader, use_gpu=torch.cuda.is_available())

    # Link prediction
    mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=False)

    # Triple Classification
    # test_dataloader.set_sampling_mode("classification")
    # acc, _ = tester.run_triple_classification()
    acc = None

    return mr, acc


def grid_search_TransE(model, transe_hyper_param_dict, dataset_name, dataset_path, valid_steps=100, early_stop_patience=4,
                       max_epochs=1000, mean_rank_lower_threshold=float("-inf")):
    valid_steps = valid_steps
    early_stop_patience = early_stop_patience

    best_mr = float("inf")
    best_acc = float("-inf")
    best_trained_model = None
    best_hyper_param = None
    best_experiment = None

    hyper_param_perm_dict = get_hyper_param_permutations(transe_hyper_param_dict)
    for experiment_index, dicct in enumerate(hyper_param_perm_dict):
        print("Start Experiment {}".format(experiment_index))
        print("- with hyper params")
        print(dicct)

        trained_model = train_Model(model, dicct, dataset_name, experiment_index, dataset_path, valid_steps,
                                     early_stop_patience, max_epochs)
        mr, acc = test_model(trained_model, dataset_path)
        # TEST
        # mr, acc = 250 - experiment_index, 0.1 * experiment_index

        print("Mean Rank: {}".format(mr))
        print("Accuracy: {}".format(acc))
        print("-----------------------------")
        if mr < best_mr:
            best_trained_model = trained_model
            best_mr = mr
            best_hyper_param = dicct
            best_experiment = experiment_index

            print("Best Mean Rank: {}".format(mr))
            print("For hyper params: {}".format(best_hyper_param))
            print("-----------------------------")

            if best_mr < mean_rank_lower_threshold:
                break

    return best_trained_model, best_hyper_param, best_experiment


def main():
    dataset_path = "../benchmarks/WN18/"
    dataset_name = "WN18"

    # Test
    valid_steps = 50
    early_stop_patience = 4
    max_epochs = 1000

    # Model and filtered Mean Rank Score from original paper
    model = TransH
    mr_score_original_paper = 303

    # Define hyper param ranges
    transe_hyper_param_dict = {}
    transe_hyper_param_dict["norm"] = [1, 2]
    transe_hyper_param_dict["margin"] = [1, 2, 5, 10]
    transe_hyper_param_dict["dimension"] = [20, 50, 100]
    transe_hyper_param_dict["learning_rate"] = [0.1, 0.01, 0.001]

    best_trained_model, best_hyper_param, best_exp = grid_search_TransE(model, transe_hyper_param_dict, dataset_name,
                                                                        dataset_path,
                                                                        valid_steps, early_stop_patience, max_epochs,
                                                                        mr_score_original_paper)

    print("-----------------------------")
    print("Best experiment was {} with hyper params:\n".format(best_exp))
    print(best_hyper_param)
    best_trained_model.save_checkpoint('../checkpoint/{}_{}_optimal_model.ckpt'.format(model.__name__, dataset_name))


if __name__ == '__main__':
    main()
