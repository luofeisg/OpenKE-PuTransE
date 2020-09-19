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
from openke.module.model import TransE
from openke.data import TrainDataLoader, IncrementalTestDataLoader
from openke.config import Trainer, Tester, Validator
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from copy import deepcopy
import time


def evolve_KG(valid_dataloader, test_dataloader, snapshot, dataset_path):
    print("Snap {}: (1) Evolve underlying KG".format(snapshot))
    print("Snap {}: (1.1) Load evaluation data".format(snapshot))
    test_dataloader.set_path(dataset_path)
    test_dataloader.evolveTripleList2(snapshot)

    print("Snap {}: (1.2) Load validation dataset".format(snapshot))
    valid_dataloader.load_snapshot(snapshot)

    print("Snap {}: (1.3) Load test dataset\n".format(snapshot))
    test_dataloader.load_snapshot(snapshot)


def train_TransE(predefined_model, hyper_param_dict, increment_path, valid_dataloader, dataset_name, snapshot,
                 valid_steps, early_stopping_patience, max_epochs):
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

    # (2) Create embedding model
    transe = predefined_model

    # Create incremental train dataloader
    train_dataloader = TrainDataLoader(
        in_path=increment_path,
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=0,
        filter_flag=0,
        neg_ent=1,
        neg_rel=0,
        random_seed=init_random_seed
    )

    # define the loss function
    model = NegativeSampling(
        model=transe,
        loss=MarginLoss(margin=margin),
        batch_size=train_dataloader.get_batch_size()
    )

    # -- validator
    validator = Validator(model=transe, data_loader=valid_dataloader)

    # -- Validation params
    bad_counts = 0
    best_hit10 = 0
    bad_count_limit_reached = False
    best_model = None

    # trainer to train the model with trainer.run(
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

        hit10 = validator.valid()

        print("hits@10 is: {}.".format(hit10))
        if hit10 > best_hit10:
            best_hit10 = hit10
            print("Best model | hit@10 of valid set is %f" % best_hit10)
            print('Save model at epoch %d.' % trained_epochs)
            best_model = deepcopy(transe)
            transe.save_checkpoint('../evaluation_framework_checkpoint/TransE_pseudoincr_{}_snapshot_{}.ckpt'
                                   .format(dataset_name, snapshot))
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


def test_model(model, test_dataloader, snapshot):
    print("Snapshot {}: Start Test Procedure at snapshot.".format(snapshot))
    tester = Tester(model=model, data_loader=test_dataloader, use_gpu=torch.cuda.is_available())

    print("Snapshot {}: Run Link Prediction.".format(snapshot))
    mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=False)

    print("Snapshot {}: Run Triple Classification.".format(snapshot))
    acc, _ = tester.run_triple_classification()

    return mr, acc


def main():
    dataset_name = "WikidataEvolve"
    dataset_path = "../benchmarks/Wikidata/{}/".format(dataset_name)

    init_random_seed = 4
    num_snapshots = 4

    print("Initial random seed is:", init_random_seed)
    print("Number of snapshots are:", num_snapshots)

    # Create directories for saving models and logging output
    checkpoint_path = Path("../evaluation_framework_checkpoint/")
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(exist_ok=True)
        print("Created folder for experiments")

    logs_path = Path("../evaluation_framework_logs/")
    if not logs_path.exists():
        logs_path.mkdir(exist_ok=True)
        print("Created folder for output logs")

    # Define training and validation variables
    early_stopping_patience = 4
    valid_steps = 100
    limit_epochs = 1000

    hyper_param_dict = {'norm': 1, 'margin': 5, 'learning_rate': 0.1, 'dimension': 100}
    norm = hyper_param_dict["norm"]
    dim = hyper_param_dict["dimension"]

    # Create incremental valid dataloader
    valid_dataloader = IncrementalTestDataLoader(in_path=dataset_path,
                                                 sampling_mode='link',
                                                 random_seed=init_random_seed,
                                                 mode='valid',
                                                 setting="incremental",
                                                 num_snapshots=num_snapshots)

    # Create incremental test dataloader
    test_dataloader = IncrementalTestDataLoader(in_path=dataset_path,
                                                sampling_mode='link',
                                                random_seed=init_random_seed,
                                                mode='test',
                                                setting="incremental",
                                                num_snapshots=num_snapshots)

    # Index TransE model with all emerging entities
    print("Index TransE model with {} global entities and {} global relations.".format(test_dataloader.entTotal,
                                                                                       test_dataloader.relTotal))
    transe = TransE(ent_tot=test_dataloader.entTotal,
                    rel_tot=test_dataloader.relTotal,
                    dim=dim,
                    p_norm=norm,
                    norm_flag=True)

    for snapshot in range(1, num_snapshots + 1):
        # (1) Evolve KG
        evolve_KG(valid_dataloader, test_dataloader, snapshot, dataset_path)

        print("Number of global entities is: {}".format(test_dataloader.entTotal))
        print("Number of global relations is: {}".format(test_dataloader.relTotal))
        print("Number of present entities is: {}".format(test_dataloader.lib.getNumCurrentlyContainedEntities()))

        increment_path = dataset_path + "pseudo_incremental/{}/".format(snapshot)
        # (2) Training
        transe = train_TransE(transe,
                     hyper_param_dict,
                     increment_path,
                     valid_dataloader,
                     dataset_name,
                     snapshot,
                     valid_steps,
                     early_stopping_patience,
                     limit_epochs)

        # (3) Evaluation
        mr, acc = test_model(transe, test_dataloader, snapshot)

        print("Mean Rank for snapshot {}: {}".format(snapshot, mr))
        print("Accuracy for snapshot {}: {}".format(snapshot, acc))


if __name__ == '__main__':
    main()
