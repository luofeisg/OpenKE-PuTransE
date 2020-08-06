from bs4 import BeautifulSoup
import re
from random import sample
from pathlib import Path
import hashlib
# from urllib2 import urlopen # Python 2
from urllib.request import urlopen  # Python 3
import bz2
import datetime
from html import unescape
import json
from qwikidata.entity import WikidataItem
from lxml import etree
import xml.etree.cElementTree as ET
from tqdm import tqdm
from datetime import datetime
from nasty_utils import DecompressingTextIOWrapper
from collections import Counter, defaultdict
from pprint import pprint
import shutil
from sklearn.model_selection import train_test_split
import random
import numpy as np


# -------------------- Output folder_structure --------------------
# dataset_<timestamp>
# |
# |- incremental
# |------- entity2id.txt    (File)
# |------- relation2id.txt  (File)
# |------- update1          (Folder)
# |----------------------------triple2id.txt (triples die zu Snapshot1 noch wahr sind) (see (3.2.1))
# |----------------------------triple-op2id.txt (alle triple operations der Phase 1) (see (3.1))
# |----------------------------train-op2id.txt alle Inserts und alle lasting Delete-Operationen aus Phase 1
# |----------------------------valid2id.txt exakt wie in static dataset - snapshot1/valid2id.txt (see (3.2.2))
# |----------------------------test2id.txt  exakt wie in static dataset - snapshot1/test2id.txt  (see (3.2.2))
# |------- update2          (Folder)
# |     ...
# |------- update<n>        (Folder)
# |
# |
# |- static
# |-------
# |------- snapshot1          (Folder)
# |----------------------------entity2id.txt    (File)
# |----------------------------relation2id.txt  (File)
# |----------------------------triple2id.txt (triples die zu Snapshot1 noch wahr sind) (see (3.2.1))
# |----------------------------
# |----------------------------train2id.txt -\          | 90%
# |----------------------------valid2id.txt ---> Split  | 10% aus triple2id.txt  (see (3.2.2))
# |----------------------------test2id.txt  -/          | 10%
# |------- snapshot2          (Folder)
# |     ...
# |------- snapshot<n>        (Folder)
# |
# |
# |- pseudo-incremental
# |------- entity2id.txt    (File)
# |------- relation2id.txt  (File)
# |------- update1          (Folder)
# |     ...
# |------- update<n>          (Folder)
# |----------------------------triple2id.txt (triples die zu Snapshot <n> wahr sind aber nicht in triple2id.txt in snap <n-1> vorkommen)
# |                            see (3.2.3)
# |----------------------------train2id.txt (alle triple aus static dataset train2id.txt außer die, die in snap <n-1> vorkommen)
# |                            see (3.2.3)
# |----------------------------valid2id.txt exakt wie in static dataset - snapshot1/valid2id.txt  (see (3.2.2))
# |----------------------------test2id.txt  exakt wie in static dataset - snapshot1/test2id.txt   (see (3.2.2))

# |------- update<n>        (Folder)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def divide_triple_operation_list(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def divide_triple_operation_list_(triple_operations_mapped, len_snap):
    return list(chunks(triple_operations_mapped, len_snap))

def write_to_file(file, triple_list):
    with file.open(mode="wt", encoding="UTF-8") as f:
        for triple in triple_list:
            subj, objc, pred = triple
            output_line = "{} {} {}\n".format(subj, objc, pred)
            f.write(output_line)


def write_to_tc_file(file, triple_list, truth_value_list):
    with file.open(mode="wt", encoding="UTF-8") as f:
        for triple, truth_value in zip(triple_list, truth_value_list):
            subj, objc, pred = triple.split()
            output_line = "{} {} {} {}\n".format(subj, objc, pred, truth_value)
            f.write(output_line)


def get_triple_operations():
    # (1) Read out sorted triple operations
    print("Load filtered triple operations.")
    compiled_triples_path = Path.cwd() / "compiled_triple_operations"
    triple_ops_file = compiled_triples_path / "compiled_triple_operations_filtered_and_sorted.txt.bz2"

    triple_operations = []
    with bz2.open(triple_ops_file, mode="rt", encoding="UTF-8") as f:
        for line in f:
            subj, objc, pred, op_type, ts = line.split()
            triple_operations.append((int(subj), int(objc), int(pred), op_type, ts))

    return triple_operations


def create_global_mapping(triple_operations, output_path):
    # (2) Map item and property ids of wikidata to new global entity and relation ids which we use in our datasets
    next_ent_id = 0
    next_rel_id = 0
    entities_dict = {}
    relations_dict = {}

    # (2.1) Iterate through triple operations and map wikidata_ids to new ids
    triple_operations_mapped = []
    static_triple_file = output_path / "mapped_triple2id.txt"
    with static_triple_file.open(mode="wt", encoding="utf-8") as out:
        for triple_op in triple_operations:
            head = triple_op[0]
            tail = triple_op[1]
            rel = triple_op[2]
            op_type = triple_op[3]
            ts = triple_op[4]

            if head not in entities_dict:
                entities_dict[head] = next_ent_id
                next_ent_id += 1
            head = entities_dict[head]

            if tail not in entities_dict:
                entities_dict[tail] = next_ent_id
                next_ent_id += 1
            tail = entities_dict[tail]

            if rel not in relations_dict:
                relations_dict[rel] = next_rel_id
                next_rel_id += 1
            rel = relations_dict[rel]

            triple_operations_mapped.append((head, tail, rel, op_type, ts))
            out.write("{} {} {}\n".format(head, tail, rel))

    # (2.2) Store entity2id and relation2id mapping
    global_ent2id_file = output_path / "entity2id.txt"
    with global_ent2id_file.open(mode="wt", encoding="utf-8") as out:
        for wikidata_id, mapped_id in entities_dict.items():
            out.write("{} {}\n".format(mapped_id, wikidata_id))

    global_rel2id_file = output_path / "relation2id.txt"
    with global_rel2id_file.open(mode="wt", encoding="utf-8") as out:
        for wikidata_id, mapped_id in relations_dict.items():
            out.write("{} {}\n".format(mapped_id, wikidata_id))

    # (2.3) Copy global entity2id and relation2id mapping to incremental and pseudo_incremental datasets
    incr_ent2id_file = output_path / "incremental" / "entity2id.txt"
    incr_rel2id_file = output_path / "incremental" / "relation2id.txt"
    shutil.copy(str(global_ent2id_file), str(incr_ent2id_file))
    shutil.copy(str(global_rel2id_file), str(incr_rel2id_file))

    ps_ent2id_file = output_path / "pseudo_incremental" / "entity2id.txt"
    ps_rel2id_file = output_path / "pseudo_incremental" / "relation2id.txt"
    shutil.copy(str(global_ent2id_file), str(ps_ent2id_file))
    shutil.copy(str(global_rel2id_file), str(ps_rel2id_file))

    return triple_operations_mapped

def sort_triple_ops_list(triple_ops_list):
    # Extract timestamps from triple_ops_list
    timestamps = []
    for operation in triple_ops_list:
        ts = operation[4]
        timestamps.append(ts)

    timestamps = np.array(timestamps)
    timestamps_sorted_indexes = timestamps.argsort().tolist()
    sorted_triple_operations = [triple_ops_list[i] for i in timestamps_sorted_indexes]

    return sorted_triple_operations

def create_directories(output_path, dir_list, num_snaps):
    directories_path_list = []
    for subdir_name in dir_list:
        sub_dir_path = output_path / subdir_name
        sub_dir_path.mkdir(exist_ok=True)
        directories_path_list.append(sub_dir_path)

        for snap in range(1, num_snaps+1):
            snap_path = sub_dir_path / str(snap)
            snap_path.mkdir(exist_ok=True)

    return directories_path_list

def create_pseudo_incremental_train_datasets(static_dataset_path, pseudo_incremental_dataset_path):
    # (6.1) Get snapshot folders from static dataset
    static_snapshot_folders = [fld for fld in static_dataset_path.iterdir() if fld.is_dir()]

    new_triple_result_set = set()
    new_train_triple_result_set = set()
    old_triple_result_set = set()

    # (6.2) Iterate static snapshot folders and substract triple set in <n> from triple set in <n-1> into pseudo_incr/triple2id.txt
    for snapshot_fld in static_snapshot_folders:
        snapshot = snapshot_fld.name

        snapshot_triple2id_file = snapshot_fld / "triple2id.txt"
        with snapshot_triple2id_file.open(mode="rt", encoding="UTF-8") as f:
            for line in f:
                subj, objc, pred = line.split()
                triple = (int(subj), int(objc), int(pred))
                new_triple_result_set.add(triple)

        # (6.3) Detect newly added triples in snapshot <n> by subtracting triple2id.txt from snapshot n - 1
        # and store into pseud_incr_dataset / <snapshot> / triple2id.txt
        pseud_incr_triple2id_set = new_triple_result_set - old_triple_result_set
        pseud_incr_triple2id_file = pseudo_incremental_dataset_path / "{}".format(snapshot) / "triple2id.txt"
        write_to_file(file=pseud_incr_triple2id_file, triple_list=pseud_incr_triple2id_set)

        # (6.4) Create train2id where only newly added train triples are covered
        # (6.4.1) Load train2id of snapshot
        snapshot_train2id_file = snapshot_fld / "train2id.txt"
        with snapshot_train2id_file.open(mode="rt", encoding="UTF-8") as f:
            for line in f:
                subj, objc, pred = line.split()
                triple = (int(subj), int(objc), int(pred))
                new_train_triple_result_set.add(triple)

        # (6.4.2) Detect newly added train triples in snapshot <n> by substracting train2id.txt from snap n with triple2id.txt of n - 1
        #         and store result into pseud_incr_dataset / <snapshot> / train2id.txt
        pseud_incr_train2id_set = new_train_triple_result_set - old_triple_result_set
        pseud_incr_train2id_file = pseudo_incremental_dataset_path / "{}".format(snapshot) / "train2id.txt"
        write_to_file(file=pseud_incr_train2id_file, triple_list=pseud_incr_train2id_set)

        old_triple_result_set = new_triple_result_set
        new_triple_result_set = set()
        new_train_triple_result_set = set()

def create_wikidata_datasets(triple_operations, num_snapshots):
    output_path = Path.cwd() / "datasets_{}".format(datetime.now().strftime("%Y_%m_%d_%H_%M"))
    output_path.mkdir(exist_ok=True)

    # (1) Create directories
    sub_directories = ["incremental", "pseudo_incremental", "static"]
    incremental_dataset_path, pseudo_incremental_dataset_path, static_dataset_path = create_directories(output_path, sub_directories, num_snapshots)

    # (2) Map item and property ids of wikidata to new global entity and relation ids which we use in our datasets
    triple_operations_mapped = create_global_mapping(triple_operations, output_path)

    # (3) Split (newly mapped) triple_operations into <num_snaps> parts and store them to triple-op2id.txt in the
    #     incremental dataset
    triple_operations_divided = divide_triple_operation_list(triple_operations_mapped, num_snapshots)

    # (4) Store triple operations for each interval to incremental dataset
    for snapshot_idx, triple_operations_list in enumerate(triple_operations_divided):
        output_lines = []
        for op in triple_operations_list:
            subj, objc, pred, op_type, ts = op
            out_line = "{} {} {} {} {}\n".format(subj, objc, pred, op_type, ts)
            # output.write(output_line + "\n")
            output_lines.append(out_line)

        # Because we count from 1
        snapshot = snapshot_idx + 1
        output_file = incremental_dataset_path / "{}".format(snapshot) / "triple-op2id.txt"
        with output_file.open(mode="wt", encoding="UTF-8") as output:
            output.writelines(output_lines)

    # (5) Generate train2id/ valid2id.txt/ test2id.txt for every snapshot. The training file will be used to learn
    # static models. Test and Validation files are constructed for all types of methods

    # Sets to track which tracks training, validation and test triples along the evolving KG
    train_triples_set = set()
    valid_triples_set = set()
    test_triples_set = set()

    # Sets which track triples which have ever been inserted into sets
    train_triples_history = set()
    valid_triples_history = set()
    test_triples_history = set()

    deleted_triple_sets = {"training_triples": [], "validation_triples": [], "test_triples": []}
    oscillated_triple_sets = {"training_triples": [], "validation_triples": [], "test_triples": []}

    # (5.1) Process triple operations for each interval to determing which triples are added and deleted
    # and which are currently true at each snapshot.

    # List which records triples which are currently true at a snapshot
    triple_result_list = []
    for snapshot_idx, triple_operations_list in enumerate(triple_operations_divided):
        # List which tracks triples which are added and deleted in interval <snapshot_idx>
        added_triples_list = []
        deleted_triples_list = []
        for op in triple_operations_list:
            subj, objc, pred, op_type, ts = op
            triple = (int(subj), int(objc), int(pred))

            if op_type == "+":
                triple_result_list.append(triple)
                added_triples_list.append(triple)
                if triple in deleted_triples_list:
                    deleted_triples_list.remove(triple)

            if op_type == "-":
                triple_result_list.remove(triple)
                if triple in added_triples_list:
                    added_triples_list.remove(triple)
                deleted_triples_list.append(triple)

        # (5.1.1) Store triple_result_list to incremental and pseudo-incremental dataset
        # (Paths: [incremental | pseudo_incremental]/<snapshot_idx+1>/triple2id.txt
        snapshot = snapshot_idx + 1
        incremental_triple2id = incremental_dataset_path / "{}".format(snapshot) / "triple2id.txt"
        static_triple2id = static_dataset_path / "{}".format(snapshot) / "triple2id.txt"
        write_to_file(file=incremental_triple2id, triple_list=triple_result_list)
        write_to_file(file=static_triple2id, triple_list=triple_result_list)

        # (5.1.2) Detect reinserts and attach them to corresponding set to ensure that triple do not jump between
        #         training, test and validation datasets
        deleted_triple_sets = {"training_triples": [], "validation_triples": [], "test_triples": []}
        negative_oscillated_triple_sets = {"training_triples": [], "validation_triples": [], "test_triples": []}
        positive_oscillated_triple_sets = {"training_triples": [], "validation_triples": [], "test_triples": []}

        excluded_triple_list = []
        for triple in added_triples_list:
            # Reinserts for train triples
            if triple in train_triples_history:
                # Attach to train set to exclude it from split
                train_triples_set.add(triple)
                excluded_triple_list.append(triple)

            if triple in valid_triples_history:
                # Attach to train set to exclude it from split
                valid_triples_set.add(triple)
                excluded_triple_list.append(triple)

            if triple in test_triples_history:
                # Attach to train set to exclude it from split
                test_triples_set.add(triple)
                excluded_triple_list.append(triple)

        added_triples_list = [triple for triple in added_triples_list if triple not in excluded_triple_list]

        # (5.1.2) Train-/Test split on newly added triples and attach them to corresponding set
        data = np.array(added_triples_list)
        # Split triple result list to 90% train and 10% eval triple
        train_triples, eval_triples = train_test_split(data, test_size=0.1, random_state=286)
        valid_triples, test_triples = train_test_split(eval_triples, test_size=0.5, random_state=286)

        # (5.1.3) Update sets and histories
        train_triples_set.update(train_triples)
        train_triples_history.update(train_triples)
        valid_triples_set.update(valid_triples)
        valid_triples_history.update(valid_triples)
        test_triples_set.update(test_triples)
        test_triples_history.update(test_triples)

        # (5.1.3) Process delete operations occurring in interval <snapshot_idx+1> in three sets
        for triple in deleted_triples_list:
            if triple in train_triples_set:
                train_triples_set.remove(triple)

            if triple in valid_triples_set:
                valid_triples.remove(triple)

            if triple in test_triples_set:
                test_triples_set.remove(triple)

        # (5.1.4) Store Train-/Valid-/Test triples to files
        static_train_file = static_dataset_path / "{}".format(snapshot) / "train2id.txt"
        write_to_file(file=static_train_file, triple_list=train_triples_set)

        static_test_file = static_dataset_path / "{}".format(snapshot) / "test2id.txt"
        write_to_file(file=static_test_file, triple_list=test_triples_set)

        incr_test_file = incremental_dataset_path / "{}".format(snapshot) / "test2id.txt"
        write_to_file(file=incr_test_file, triple_list=test_triples_set)

        pseudo_incr_test_file = pseudo_incremental_dataset_path / "{}".format(snapshot) / "test2id.txt"
        write_to_file(file=pseudo_incr_test_file, triple_list=test_triples_set)

        static_valid_file = static_dataset_path / "{}".format(snapshot) / "valid2id.txt"
        write_to_file(file=static_valid_file, triple_list=valid_triples_set)

        incr_valid_file = incremental_dataset_path / "{}".format(snapshot) / "valid2id.txt"
        write_to_file(file=incr_valid_file, triple_list=valid_triples_set)

        pseudo_incr_valid_file = pseudo_incremental_dataset_path / "{}".format(snapshot) / "valid2id.txt"
        write_to_file(file=pseudo_incr_valid_file, triple_list=valid_triples_set)

    # (6) Create triple2id.txt and train2id.txt for pseudo_incremental dataset
    create_pseudo_incremental_train_datasets(static_dataset_path, pseudo_incremental_dataset_path)

    # (7) Create train-op2id.txt
    new_train_triple_set = set()
    old_train_triple_set = set()
    for snapshot, triple_operations_list in enumerate(triple_operations_divided):
        # Load train triples from snapshot <snapshot_idx>
        snapshot_train2id_file = static_dataset_path / "{}".format(snapshot) / "train2id.txt"
        with snapshot_train2id_file.open(mode="rt", encoding="UTF-8") as f:
            for line in f:
                subj, objc, pred = line.split()
                triple = (int(subj), int(objc), int(pred))
                new_train_triple_set.add(triple)

        # Determine inserts and deletes
        inserted_train_triples = new_train_triple_set - old_train_triple_set
        deleted_train_triples = old_train_triple_set - new_train_triple_set

        # Gather timestamps for these triples from triple_operations_list
        train_triple_operations = []
        for operation in triple_operations_list:
            subj, objc, pred, op_type, ts = operation
            triple = (int(subj), int(objc), int(pred))

            if triple in inserted_train_triples and op_type == "+":
                train_triple_operations.append((subj, objc, pred, op_type, ts))
            if triple in deleted_train_triples and op_type == "-":
                train_triple_operations.append((subj, objc, pred, op_type, ts))

        # sort list of triple operations
        train_triple_operations = sort_triple_ops_list(train_triple_operations)

        incr_triple_op2id_file = incremental_dataset_path / "{}".format(snapshot) / "train-op2id.txt"
        with incr_triple_op2id_file.open(mode="wt", encoding="UTF-8") as output:
            for triple_op in train_triple_operations:
                output_line = "{} {} {} {}\n".format(triple_op[0], triple_op[1], triple_op[2], triple_op[3])
                output.write(output_line)

        old_train_triple_set = new_train_triple_set
        new_train_triple_set = set()


def detect_and_store_deleted_and_oscillating_triples(output_path, num_snapshots, dataset="train"):
    # (8) Detect reinserts and (positive | negative) oscillated triples for train, valid, test triples
    # Status Transitions:
    # (0)->(1) Inserted
    # (1)->(2) Deleted
    # (2)->(3) Positive Oscillated
    # (3)->(4) Negative Oscillated
    # (4)->(3) Positive Oscillated

    static_dataset_path = output_path / "static"
    triples_status = {}

    deleted_triples_set = set()
    negative_oscillated_triples_set = set()
    positive_oscillated_triples_set = set()

    new_triple_set = set()
    old_triple_set = set()
    for snapshot in range(1, num_snapshots + 1):
        # (8.1) Load train triples from snapshot <snapshot_idx>
        snapshot_file = static_dataset_path / "{}".format(snapshot) / "{}2id.txt".format(dataset)
        with snapshot_file.open(mode="rt", encoding="UTF-8") as f:
            for line in f:
                subj, objc, pred = line.split()
                triple = (int(subj), int(objc), int(pred))
                new_triple_set.add(triple)

        # (8.2) Determine inserts and deletes
        inserts = new_triple_set - old_triple_set
        deletes = old_triple_set - new_triple_set

        # Switch old and new triple set for next snapshot
        old_triple_set = new_triple_set
        new_triple_set = set()

        # (8.3) Detect Deletes and Oscillating triples
        for triple in inserts:
            if triple in triples_status:
                status = triples_status[triple]

                if status == "Deleted":
                    status = "Positive Oscillated"
                    deleted_triples_set.remove(triple)
                    positive_oscillated_triples_set.add(triple)

                if status == "Negative Oscillated":
                    status = "Positive Oscillated"
                    negative_oscillated_triples_set.remove(triple)
                    positive_oscillated_triples_set.add(triple)

                triples_status[triple] = status
            else:
                triples_status[triple] = "Inserted"

        for triple in deletes:
            status = triples_status[triple]

            if status == "Inserted":
                status = "Deleted"
                deleted_triples_set.add(triple)

            if status == "Positive Oscillated":
                status = "Negative Oscillated"
                positive_oscillated_triples_set.remove(triple)
                negative_oscillated_triples_set.add(triple)

            triples_status[triple] = status

        # (8.4) Store deleted and oscillated triples to files
        incremental_dataset_path = output_path / "incremental"
        snapshot_folder = incremental_dataset_path / "{}".format(snapshot)

        deleted_triple_file = snapshot_folder / "tc_negative_deleted_{}_triples.txt".format(dataset)
        positive_oscillated_file = snapshot_folder / "tc_positive_oscillated_{}_triples.txt".format(dataset)
        negative_oscillated_file = snapshot_folder / "tc_negative_oscillated_{}_triples.txt".format(dataset)

        write_to_tc_file(file=deleted_triple_file, triple_list=deleted_triples_set,
                         truth_value_list=[0] * len(deleted_triples_set))

        write_to_tc_file(file=negative_oscillated_file, triple_list=negative_oscillated_triples_set,
                         truth_value_list=[0] * len(negative_oscillated_triples_set))

        write_to_tc_file(file=positive_oscillated_file, triple_list=positive_oscillated_triples_set,
                         truth_value_list=[1] * len(positive_oscillated_triples_set))

def main():
    num_snapshots = 5
    triple_operations = get_triple_operations()
    create_wikidata_datasets(triple_operations, num_snapshots)

if __name__ == '__main__':
    main()

    # Load all triple operations to detect delete operations for added train triple
    # We perform a randomly train test valid split in every snapshot for large proportions to training data (90%) which is
    # a complex task. Due to the fact that our evaluations is based on a evolving set of triples rather than a static
    # set of triple it is a complex task to produce training and test data for every snapshot. There are many options to
    # design datasets for every snapshot. On the one hand we could grow a test and evaluation dataset simultaneously.
    # This means we divide triple operations in specific proportions for exaamples let say 90 10
    # In this case we would demand from a model which was learned by set of training data to predict
    # truth values from a seperated set of evaluation triple operations. For a static set this task is more simple, since we only
    # consider train and test triples which are true at a given time. In case of the stated example we would also include
    # delete operations; that means we have to deal with negative triples at a given time.
    # The ability to incorporate delete operations by a incremental model is an important aspect, since examples like
    # WIkidata show that there a high number of deletes which ocurred since the KG is existing.
    # Since a link prediction on deleted i.e. negative triple to obtain the MR or hits@10 doesnt make sense one could rely
    # on a triple classification task to say whether a model can handle deleted triple. But by restricting this evaluation
    # solely to triple operations sourced in the evaluation datastet would also make no sense and quite unfair for the model.
    # In the first place in a static setting models are tested to predict the truth value of triples which have not been
    # seen by a model.
    # But to demand from a model to predict it as true if it is existing in a KG and then predict it as
    # false at a later stage although it has not seen a insert but moreover a delete operation before would be quite unfair.
    # In an incremental setting which has up to this point not addressed in the literature
    # we therefore point out to check if an incremental model to unlearn facts which have been seen, i.e.
    # training triples before we check it is able to predict the truth value of evaluation triples affected by unseen operations
    # correctly.
    # In this point we address both triple operations seen and unseen by the model
    #
    # Regarding train data we want to hold the evolving training dataset consistent. In other words, we want to track
    # if a triple was added to the train dataset and seen by the model and also if it was deleted in the meantime to
    # enable the model to forget.
    # We derive newly added training triples by loading the train2id files in the pseudo-incremental folders since these files
    # are created by only considering triples which have been inserted in the correspoding period.
    # Detecting delete operations however is a more complex task since a training triple of a previous snapshot can be moved
    # to the evaluation datasets. I.e. the triple is still contained in the KG and not removed from the train dataset.

    # To allow an incremental model to recognize deleted of previously seen train operations therfore we provide a train
    # dataset of triple operations containing newly added triples in a phase n and all delete operations in a phase to
    # enable him to detect a delete operation in its dataset.
    # To allow an incremental model to recognize deleted of previously seen train operations we therefore track which
    # triples emerged in a train dataset at a snapshot so far and traverse all tripleoperations of the current snaphot at
    # hand to detect corresponding delete operations

    # (4) Generate evaluations datasets for deleted and oscilatted triples if available. We gather them from both train2id.txt
    #     and triple2id.txt. Thus we have consider two degress of difficulty in the test phase. The deleted triples we
    #     gather from the train dataset are known by the model whereas the deleted triples in the triple2id.txt may be unknown

