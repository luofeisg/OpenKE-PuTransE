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

from openke.data import IncrementalTrainDataLoader, IncrementalTestDataLoader
from openke.config import Parallel_Universe_Config
from openke.module.model import TransE
from datetime import datetime

def evolve_KG(PuTransX_model, snapshot):
    print("Snap {}: (1) Evolve underlying KG".format(snapshot))
    print("Snap {}: (1.1) Evolve training dataset".format(snapshot))
    PuTransX_model.train_dataloader.load_snapshot(snapshot)

    print("Snap {}: (1.2) Load evaluation data".format(snapshot))
    # PuTransX_model.data_loader.evolveTripleList(snapshot)
    PuTransX_model.data_loader.evolveTripleList2(snapshot)

    print("Snap {}: (1.2.3) Load validation dataset".format(snapshot))
    PuTransX_model.valid_dataloader.load_snapshot(snapshot)

    print("Snap {}: (1.2.3) Load test dataset\n".format(snapshot))
    PuTransX_model.data_loader.load_snapshot(snapshot)

def PuTransX_training_procedure(PuTransX_model, snapshot, embedding_spaces):
    print("Snap {}: Start training procedure.\n".format(snapshot))
    PuTransX_model.training_identifier = "snapshot_{}_6000_spaces".format(snapshot)
    PuTransX_model.reset_valid_variables()

    print("Snap {}: (1) Check if best model already has been trained for snaphot.".format(snapshot))
    best_model_filename = "Best_model_Pu{}_snapshot_{}_6000_spaces.ckpt".format(PuTransX_model.embedding_model.__name__, snapshot)
    best_model_file = Path(PuTransX_model.checkpoint_dir) / best_model_filename

    # If best model does not exist yet, train it
    if not best_model_file.exists():
        print("Snap {}: (1.1) Best model does not exists yet. Train PuTransE with limit of {} universes.\n".format(snapshot, embedding_spaces))
        PuTransX_model.train_parallel_universes(embedding_spaces)
    else:
        print("Snap {}: Model already trained\n")

    print("Snap {}: Load best model...")
    PuTransX_model.load_parameters(best_model_filename)
    print("Snap {}: Loaded best model with {} trained universes.\n".format(snapshot, PuTransX_model.next_universe_id))


def PuTransX_evaluation_procedure(PuTransX_model, incremental_strategy, snapshot):
    PuTransX_model.incremental_strategy = incremental_strategy
    print("Snap {}: Start evaluation procedure.".format(snapshot))
    print("-- incremental strategy: {}.".format(incremental_strategy))
    print("-- Timestamp: {}.\n".format(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')))

    print("Snap {}: (1) Run Evaluation with missing energy score handling: {}".format(snapshot, "Infinity Score Handling"))
    PuTransX_model.missing_embedding_handling = "last_rank"

    print("\nSnap {}: (1.1) Run Link Prediction...".format(snapshot))
    PuTransX_model.run_link_prediction()

    print("\nSnap {}: (1.2) Run Triple Classification + Negative Triple Classifcation...".format(snapshot))
    PuTransX_model.run_triple_classification_from_files(snapshot)

    print("\nSnap {}: (2) Run Evaluation with missing energy score handling: {}.".format(snapshot, "Null Vector Handling"))
    PuTransX_model.missing_embedding_handling = "null_vector"

    print("\nSnap {}: (2.1) Run Link Prediction...".format(snapshot))
    PuTransX_model.run_link_prediction()

    print("\nSnap {}: (2.2) Run Triple Classification + Negative Triple Classifcation...".format(snapshot))
    PuTransX_model.run_triple_classification_from_files(snapshot)

    print("\nSnap {}: Finished evaluation procedure at: {}.\n".format(snapshot, datetime.now().strftime('%Y-%m-%dT%H:%M:%S')))
    PuTransX_model.reset_evaluation_helpers()

def main():
    checkpoint_path = Path("../evaluation_framework_checkpoint/")
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(exist_ok=True)
        print("Created folder for experiments")

    logs_path = Path("../evaluation_framework_logs/")
    if not logs_path.exists():
        logs_path.mkdir(exist_ok=True)
        print("Created folder for output logs")

    evaluated_markers_path = checkpoint_path / "evaluated_marker"
    if not evaluated_markers_path.exists():
        evaluated_markers_path.mkdir(exist_ok=True)
        print("Created folder for evaluated markers")

    init_random_seed = 4
    num_snapshots = 4
    incremental_dataset_path = "../benchmarks/Wikidata/WikidataEvolve/"

    early_stopping_patience = 5
    valid_steps = 200
    limit_embedding_spaces = 6000
    print("Initial random seed is:", init_random_seed)
    print("Number of snapshots are:", num_snapshots)

    # Create incremental train dataloader
    train_dataloader = IncrementalTrainDataLoader(
        in_path=incremental_dataset_path,
        nbatches=20,
        threads=8,
        sampling_mode="normal",
        bern_flag=0,
        filter_flag=0,
        neg_ent=1,
        neg_rel=0,
        random_seed=init_random_seed,
        incremental_setting=True,
        num_snapshots=num_snapshots)

    # Create incremental valid dataloader
    valid_dataloader = IncrementalTestDataLoader(in_path=incremental_dataset_path,
                                                sampling_mode='link',
                                                random_seed=init_random_seed,
                                                mode='valid',
                                                setting="incremental",
                                                num_snapshots=num_snapshots)

    # Create incremental test dataloader
    test_dataloader = IncrementalTestDataLoader(in_path=incremental_dataset_path,
                                                sampling_mode='link',
                                                random_seed=init_random_seed,
                                                mode='test',
                                                setting="incremental",
                                                num_snapshots=num_snapshots)

    # -> Set parameters for model used in the Parallel Universe Config (in this case TransE)
    embedding_method = TransE
    param_dict = {
        'dim': 20,
        'p_norm': 1,
        'norm_flag': 1
    }

    # Create Parallel Universe configuration
    PuTransE = Parallel_Universe_Config(
        training_identifier='',
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,

        # hyper parameter ranges
        min_margin=1,
        max_margin=5,
        min_lr=0.001,
        max_lr=0.1,
        min_num_epochs=50,
        max_num_epochs=200,
        min_triple_constraint=500,
        max_triple_constraint=1500,
        min_balance=0.25,
        max_balance=0.5,

        # embedding method
        embedding_model=embedding_method,
        embedding_model_param=param_dict,

        checkpoint_dir="../evaluation_framework_checkpoint/",
        valid_steps=valid_steps,
        early_stopping_patience=early_stopping_patience,

        save_steps=None,
        training_setting="incremental",
        missing_embedding_handling="last_rank")

    for snapshot in range(1, num_snapshots + 1):
        # Configure log for snapshot
        # orig_sys_stdout = sys.stdout
        # snapshot_log = logs_path / "PuTransE_on_WikidataEvolve_snapshot_{}_on_{}.txt".format(snapshot, datetime.now().strftime('%Y_%m_%d'))
        # log = snapshot_log.open(mode="wt", encoding="UTF-8")
        # sys.stdout = log

        # (1) Evolve underlying KG
        evolve_KG(PuTransE, snapshot)

        # (2) Training
        PuTransX_training_procedure(PuTransE, snapshot, limit_embedding_spaces)

        # (3) Evaluation
        already_evaluated_marker_file = evaluated_markers_path / "PuTransE_snapshot_{}_6000_spaces.evaluated".format(snapshot)
        if not already_evaluated_marker_file.exists():
            PuTransX_evaluation_procedure(PuTransE, "normal", snapshot)
            PuTransX_evaluation_procedure(PuTransE, "deprecate", snapshot)
            already_evaluated_marker_file.touch()
        else:
            print("Model has already been evaluated on snapshot {}.".format(snapshot))
            print("Continue to next snapshot! ...")

        # Close log file
        # sys.stdout = orig_sys_stdout
        # snapshot_log.close()

if __name__ == '__main__':
    main()