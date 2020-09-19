'''
MIT License

Copyright (c) 2020 Rashid Lafraie

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import sys
from pathlib import Path
import torch

openke_path = Path.cwd().parents[0]
print(openke_path)
sys.path.append(str(openke_path))

# OpenKE modules
import openke
from openke.module.model import TransE
from openke.data import TrainDataLoader, TestDataLoader
from openke.config import Trainer, Tester, Validator
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from copy import deepcopy
import time


def get_hyper_param_permutations(transe_hyper_param_dict):
    permutated_hyperparam_dict = [{"norm": norm, "margin": margin, "learning_rate": lr, "dimension": dim}
                                  for norm in transe_hyper_param_dict["norm"]
                                  for margin in transe_hyper_param_dict["margin"]
                                  for lr in transe_hyper_param_dict["learning_rate"]
                                  for dim in transe_hyper_param_dict["dimension"]
                                  ]

    return permutated_hyperparam_dict


def train_TransE(hyper_param_dict, dataset_name, experiment_index, dataset_path, valid_steps, early_stop_patience,
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
        bern_flag=0,
        filter_flag=0,
        neg_ent=1,
        neg_rel=0,
        random_seed=init_random_seed)

    # (2) Create embedding model
    transe = TransE(
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
    valid_dl = TestDataLoader(train_dataloader.in_path, "link", mode='valid', load_all_triples=True)
    validator = Validator(model=transe, data_loader=valid_dl)

    # -- Validation params
    valid_steps = valid_steps
    early_stopping_patience = early_stop_patience
    bad_counts = 0
    best_hit10 = 0
    bad_count_limit_reached = False
    best_model = None

    # trainer to train the model with trainer.run()
    trainer = Trainer(model=model, data_loader=train_dataloader, alpha=lr, train_times=valid_steps,
                      use_gpu=torch.cuda.is_available())

    trained_epochs = 0
    training_duration = 0
    while not bad_count_limit_reached and trained_epochs < max_epochs:
        # Train and measure training time
        start_time = time.time()
        trainer.run()
        end_time = time.time()
        training_duration = training_duration + (end_time - start_time)

        trained_epochs += valid_steps

        # Validation
        hit10 = validator.valid()

        print("hits@10 is: {}.".format(hit10))
        if hit10 > best_hit10:
            best_hit10 = hit10
            print("Best model | hit@10 of valid set is %f" % best_hit10)
            print('Save model at epoch %d.' % trained_epochs)
            best_model = deepcopy(transe)
            transe.save_checkpoint(
                '../evaluation_framework_checkpoint/transe_{}_exp{}.ckpt'.format(dataset_name, experiment_index))
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

    print('Time took for training: {:5.3f}s'.format(training_duration), end='\n')

    return best_model


def test_model(model, dataset_path):
    test_dataloader = TestDataLoader(dataset_path, "link", mode='test', load_all_triples=True)
    tester = Tester(model=model, data_loader=test_dataloader, use_gpu=torch.cuda.is_available())

    # Link prediction
    mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=False)

    # Triple Classification
    test_dataloader.set_sampling_mode("classification")
    acc, _ = tester.run_triple_classification()

    return mr, acc


def grid_search_TransE(transe_hyper_param_dict, dataset_name, dataset_path, valid_steps=100, early_stop_patience=4,
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

        trained_model = train_TransE(dicct, dataset_name, experiment_index, dataset_path, valid_steps,
                                     early_stop_patience, max_epochs)
        mr, acc = test_model(trained_model, dataset_path)

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
    snapshot = 3

    dataset_path = "../benchmarks/Wikidata/WikidataEvolve/static/"
    dataset_snapshot_path = dataset_path + "{}/".format(snapshot)
    dataset_name = "WikidataEvolve"

    early_stop_patience = 4
    valid_steps = 100
    max_epochs = 1000

    # Define hyper param ranges
    best_hyper_param = {'norm': 1, 'margin': 5, 'learning_rate': 0.1, 'dimension': 100}

    print("=====")
    print(" Start snapshot {}\n".format(snapshot))


    trained_model = train_TransE(best_hyper_param,
                                 dataset_name,
                                 "_snapshot{}_without_grid_search".format(snapshot),
                                 dataset_snapshot_path,
                                 valid_steps,
                                 early_stop_patience,
                                 max_epochs)

    mr, acc = test_model(trained_model, dataset_snapshot_path)
    print("Mean Rank for snapshot {}: {}".format(snapshot, mr))
    print("Accuracy for snapshot {}: {}".format(snapshot, acc))


if __name__ == '__main__':
    main()
