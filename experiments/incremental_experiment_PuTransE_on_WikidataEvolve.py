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
    print("Snap {}: (1) Evolve underlying KG\n".format(snapshot))
    print("Snap {}: (1.1) Evolve training dataset\n".format(snapshot))
    PuTransX_model.train_dataloader.load_snapshot(snapshot)

    print("Snap {}: (1.2) Load evaluation data\n".format(snapshot))
    PuTransX_model.data_loader.evolveTripleList(snapshot)

    print("Snap {}: (1.2.3) Load validation dataset\n".format(snapshot))
    PuTransX_model.valid_dataloader.load_snapshot(snapshot)

    print("Snap {}: (1.2.3) Load test dataset\n".format(snapshot))
    PuTransX_model.data_loader.load_snapshot(snapshot)

def PuTransX_training_procedure(PuTransX_model, snapshot, embedding_spaces):
    print("Snap {}: Start training procedure.\n".format(snapshot))
    PuTransX_model.training_identifier = "snapshot{}".format(snapshot)
    PuTransX_model.reset_valid_variables()

    print("Snap {}: (1) Check if best model already has been trained for snaphot.\n".format(snapshot))
    best_model_filename = "Best_model_Pu{}_snapshot{}.ckpt".format(PuTransX_model.embedding_model.__name__, snapshot)
    best_model_file = Path(PuTransX_model.checkpoint_dir) / best_model_filename

    # If best model does not exist yet, train it
    if not best_model_file.exists():
        print("Snap {}: (1.1) Best model does not exists yet. Train PuTransE with limit of {} universes.\n".format(snapshot, embedding_spaces))
        PuTransX_model.train_parallel_universes(embedding_spaces)
    else:
        print("Snap {}: Model already trained\n")

    print("Snap {}: Load best model...\n")
    PuTransX_model.load_parameters(best_model_filename)
    print("Snap {}: Loaded best model with {} trained universes.\n".format(snapshot, PuTransX_model.next_universe_id))


def PuTransX_evaluation_procedure(PuTransX_model, incremental_strategy, snapshot):
    PuTransX_model.incremental_strategy = incremental_strategy
    print("Snap {}: Start evaluation procedure.\n".format(snapshot))
    print("-- incremental strategy: {}.\n".format(incremental_strategy))
    print("-- Timestamp: {}.\n".format(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')))

    print("Snap {}: (1) Conduct evaluation setting with missing energy score handling: {}\n".format(snapshot, "Infinity Score Handling"))
    PuTransX_model.missing_embedding_handling = "last_rank"

    print("Snap {}: (1.1) Run Link Prediction...\n".format(snapshot))
    # PuTransX_model.run_link_prediction()

    print("Snap {}: (1.2) Run Triple Classification + Negative Triple Classifcation...\n".format(snapshot))
    PuTransX_model.run_triple_classification_from_files(snapshot)

    print("Snap {}: (2) Run Link Prediction with mode: {}.\n".format(snapshot, "Null Vector Handling"))
    PuTransX_model.missing_embedding_handling = "null_vector"

    print("Snap {}: (2.1) Run Link Prediction...\n".format(snapshot))
    # PuTransX_model.run_link_prediction()

    print("Snap {}: (2.2) Run Triple Classification + Negative Triple Classifcation...\n".format(snapshot))
    PuTransX_model.run_triple_classification_from_files(snapshot)

    print("Snap {}: Finished evaluation procedure at: {}.\n".format(snapshot, datetime.now().strftime('%Y-%m-%dT%H:%M:%S')))
    PuTransX_model.reset_evaluation_helpers()

def main():
    checkpoint_path = Path("../evaluation_framework_checkpoint/")
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(exist_ok=True)
        print("Created folder for experiments")

    init_random_seed = 4
    num_snapshots = 4
    incremental_dataset_path = "../benchmarks/Wikidata/WikidataEvolve/"

    early_stopping_patience = 5
    valid_steps = 100 # TODO set to 100
    limit_embedding_spaces = 100
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

        const_num_epochs=10,  # TODO Remove -> set only for test

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
        # (1) Evolve underlying KG
        evolve_KG(PuTransE, snapshot)

        # (2) Training
        PuTransX_training_procedure(PuTransE, snapshot, limit_embedding_spaces)

        # (3) Evaluation
        PuTransX_evaluation_procedure(PuTransE, "normal", snapshot)
        PuTransX_evaluation_procedure(PuTransE, "deprecate", snapshot)

if __name__ == '__main__':
    main()