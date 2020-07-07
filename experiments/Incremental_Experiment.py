from openke.data import IncrementalTrainDataLoader, TrainDataLoader
from openke.config import Parallel_Universe_Config
from openke.module.model import TransE, TransH
from pathlib import Path
import torch

if __name__ == '__main__':
    # Initialize random seed to make experiments reproducable
    # init_random_seed = randint(0, 2147483647)
    init_random_seed = 4
    num_snapshots = 5
    print("Initial random seed is:", init_random_seed)

    # create incremental traindataloader
    train_dataloader = IncrementalTrainDataLoader(
        in_path="../../benchmarks/Wikidata/datasets/",
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=0,
        neg_ent=1,
        neg_rel=0,
        random_seed=init_random_seed,
        incremental_setting=True,
        num_snapshots=5)

    # Initialize Parallel Universe configuration

    # -> Set parameters for model used in the Parallel Universe Config (in this case TransE)
    param_dict = {
        'dim': 50,
        'p_norm': 1,
        'norm_flag': 1
    }

    embedding_method = TransE

    # PuTransE = Parallel_Universe_Config(
    #     training_identifier='',
    #     train_dataloader=train_dataloader,
    #     test_dataloader=None,
    #     initial_num_universes=5000,
    #     min_margin=1,
    #     max_margin=4,
    #     min_lr=0.01,
    #     max_lr=0.1,
    #     min_num_epochs=50,
    #     max_num_epochs=200,
    #     min_triple_constraint=500,
    #     max_triple_constraint=2000,
    #     min_balance=0.25,
    #     max_balance=0.5,
    #     embedding_model=embedding_method,
    #     embedding_model_param=param_dict,
    #     checkpoint_dir="./checkpoint/",
    #     valid_steps=100,
    #     save_steps=1000)
# def load_snapshot(snapshot_idx):
#     PuTransE.train_dataloader.load_snaphot(snapshot_idx)
#     PuTransE.test_dataloader.load_snaphot(snapshot_idx)


    for snapshot in range(1, num_snapshots+1):
#         # Train and test incremental model PuTransE
          train_dataloader.load_snapshot(snapshot)
#           # PuTransE.training_identifier = "snapshot1_normal_strategy"
#           # PuTransE.training_identifier = "snapshot1_deprecated_strategy"
#           # Save models

#         # Train and test Static models
            # Train PuTransE
#           # Train TransE
#           # Save models
#
#         # create testdataloader(snap)
#
#         # test
#         # link prediction
#         # triple classification
#
#         # write results into
#
#         # testdataloader = None
#
#     # Train static


