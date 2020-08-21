from pathlib import Path
import sys

openke_path = Path.cwd().parents[0]
print(openke_path)
sys.path.append(str(openke_path))

from openke.config import Parallel_Universe_Config
from openke.data import TrainDataLoader, TestDataLoader
from openke.module.model import RESCAL
import torch

if __name__ == '__main__':
    # Initialize random seed to make experiments reproducable
    # init_random_seed = randint(0, 2147483647)
    init_random_seed = 4

    print("Initial random seed is:", init_random_seed)

    # Initialize TrainDataLoader for sampling of examples
    train_dataloader = TrainDataLoader(
        in_path="../benchmarks/WN18/",
        nbatches=20,
        threads=8,
        sampling_mode="normal",
        bern_flag=0,
        filter_flag=0,
        neg_ent=1,
        neg_rel=0,
        random_seed=init_random_seed)

    # Initialize TestDataLoader which provides test data
    # test_dataloader = TestDataLoader(train_dataloader.in_path, "link")
    test_dataloader = TestDataLoader(train_dataloader.in_path, "link")

    # Set parameters for model used in the Parallel Universe Config (in this case RESCAL)
    param_dict = {
        'dim': 20
    }

    embedding_method = RESCAL

    PuRESCAL = Parallel_Universe_Config(
        training_identifier='',
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        initial_num_universes=None,
        min_margin=1,
        max_margin=4,
        min_lr=0.001,
        max_lr=0.1,
        min_num_epochs=50,
        max_num_epochs=200,
        min_triple_constraint=500,
        max_triple_constraint=2000,
        min_balance=0.25,
        max_balance=0.5,
        embedding_model=embedding_method,
        embedding_model_param=param_dict,
        checkpoint_dir="../checkpoint/",
        valid_steps=100,
        save_steps=10000,
        training_setting="static",
        incremental_strategy=None)

    PuRESCAL.train_parallel_universes(6000)
    PuRESCAL.run_link_prediction()
