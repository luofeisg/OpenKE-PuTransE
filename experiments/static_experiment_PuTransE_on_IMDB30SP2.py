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

from pathlib import Path
import sys

openke_path = Path.cwd().parents[0]
print(openke_path)
sys.path.append(str(openke_path))

from openke.config import Parallel_Universe_Config
from openke.data import TrainDataLoader, TestDataLoader
from openke.module.model import TransE, TransH
import random


if __name__ == '__main__':
    init_random_seed = random.randint(0, 2147483647)
    print("Initial random seed is:", init_random_seed)

    # Initialize TrainDataLoader for sampling of examples
    train_dataloader = TrainDataLoader(
        in_path="../benchmarks/IMDB-30SP/snapshot2/",
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

    # Set parameters for model used in the Parallel Universe Config (in this case TransE)
    param_dict = {
        'dim': 20,
        'p_norm': 1,
        'norm_flag': 1
    }

    embedding_method = TransE

    PuTransE = Parallel_Universe_Config(
        training_identifier='PuTransE_IMDB30SP2',
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

    PuTransE.train_parallel_universes(6000)
    PuTransE.run_link_prediction()
