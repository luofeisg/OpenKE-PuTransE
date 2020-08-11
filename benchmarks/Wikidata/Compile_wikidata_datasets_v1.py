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
import operator


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


def write_to_file(file, triples_iterable):
    with file.open(mode="wt", encoding="UTF-8") as f:
        for triple in triples_iterable:
            subj, objc, pred = triple
            output_line = "{} {} {}\n".format(subj, objc, pred)
            f.write(output_line)


def write_to_tc_file(file, triple_list, truth_value_list):
    with file.open(mode="wt", encoding="UTF-8") as f:
        for triple, truth_value in zip(triple_list, truth_value_list):
            subj, objc, pred = triple
            output_line = "{} {} {} {}\n".format(subj, objc, pred, truth_value)
            f.write(output_line)


def get_compressed_triple_operations(triple_operations_file):
    # (1) Read out sorted triple operations
    triple_operations = []
    with bz2.open(triple_operations_file, mode="rt", encoding="UTF-8") as f:
        for line in f:
            subj, objc, pred, op_type, ts = line.split()
            triple_operations.append((int(subj), int(objc), int(pred), op_type, ts))

    return triple_operations

def get_triple_operations(triple_operations_file):
    # (1) Read out sorted triple operations
    triple_operations = []
    with bz2.open(triple_operations_file, mode="rt", encoding="UTF-8") as f:
        for line in f:
            subj, objc, pred, op_type, ts = line.split()
            triple_operations.append((int(subj), int(objc), int(pred), op_type, ts))

    return triple_operations


def create_global_mapping(triple_operations, output_path, dataset_paths_dict):
    # (2) Map item and property ids of wikidata to new global entity and relation ids which we use in our datasets
    next_ent_id = 0
    next_rel_id = 0
    entities_dict = {}
    relations_dict = {}

    # (2.1) Iterate through triple operations and map wikidata_ids to new ids
    triple_operations_mapped = []
    static_triple_file = output_path / "mapped_triple-op2id.txt"
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
            out.write("{} {} {} {} {}\n".format(head, tail, rel, op_type, ts))

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
    incr_ent2id_file = dataset_paths_dict["incremental_dataset_path"] / "entity2id.txt"
    incr_rel2id_file = dataset_paths_dict["incremental_dataset_path"] / "relation2id.txt"
    shutil.copy(str(global_ent2id_file), str(incr_ent2id_file))
    shutil.copy(str(global_rel2id_file), str(incr_rel2id_file))

    ps_ent2id_file = dataset_paths_dict["pseudo_incremental_dataset_path"] / "entity2id.txt"
    ps_rel2id_file = dataset_paths_dict["pseudo_incremental_dataset_path"] / "relation2id.txt"
    shutil.copy(str(global_ent2id_file), str(ps_ent2id_file))
    shutil.copy(str(global_rel2id_file), str(ps_rel2id_file))

    return triple_operations_mapped


def sort_triple_ops_list(triple_ops_list):
    sorted_triple_operations = sorted(triple_ops_list, key=operator.itemgetter(4, 0, 1, 2, 3))

    # # Extract timestamps from triple_ops_list
    # timestamps = []
    # for operation in triple_ops_list:
    #     ts = operation[4]
    #     timestamps.append(ts)
    #
    # timestamps = np.array(timestamps)
    # timestamps_sorted_indexes = timestamps.argsort().tolist()
    # sorted_triple_operations = [triple_ops_list[i] for i in timestamps_sorted_indexes]

    return sorted_triple_operations


def create_directories(output_path, num_snaps):
    subdir_names = ["static_dataset_path", "incremental_dataset_path", "pseudo_incremental_dataset_path"]
    paths_dict = {}

    for name in subdir_names:
        sub_dir_path = output_path / name
        sub_dir_path.mkdir(exist_ok=True)
        paths_dict[name] = sub_dir_path

        for snap in range(1, num_snaps + 1):
            snap_path = sub_dir_path / str(snap)
            snap_path.mkdir(exist_ok=True)

    return paths_dict


def create_pseudo_incremental_train_datasets(paths, num_snapshots):
    # (6.1) Get snapshot folders from static dataset
    static_dataset_path = paths["static_dataset_path"]
    pseudo_incremental_dataset_path = paths["pseudo_incremental_dataset_path"]

    new_triple_result_set = set()
    new_train_triple_result_set = set()
    old_triple_result_set = set()

    # (6.2) Iterate static snapshot folders and substract triple set in <n> from triple set in <n-1> into pseudo_incr/triple2id.txt
    for snapshot in range(1, num_snapshots + 1):
        new_triple_result_set = load_snapshot_triple_set(static_dataset_path, snapshot)

        # (6.3) Detect newly added triples in snapshot <n> by subtracting triple2id.txt from snapshot n - 1
        # and store into pseud_incr_dataset / <snapshot> / triple2id.txt
        pseud_incr_triple2id_set = new_triple_result_set - old_triple_result_set
        pseud_incr_triple2id_file = pseudo_incremental_dataset_path / "{}".format(snapshot) / "triple2id.txt"
        write_to_file(file=pseud_incr_triple2id_file, triples_iterable=pseud_incr_triple2id_set)

        # (6.4) Create train2id where only newly added train triples are covered
        # (6.4.1) Load train2id of snapshot
        new_train_triple_result_set = load_snapshot_triple_set(static_dataset_path, snapshot, filename="train2id.txt")

        # (6.4.2) Detect newly added train triples in snapshot <n> by substracting train2id.txt from snap n with triple2id.txt of n - 1
        #         and store result into pseud_incr_dataset / <snapshot> / train2id.txt
        pseud_incr_train2id_set = new_train_triple_result_set - old_triple_result_set
        pseud_incr_train2id_file = pseudo_incremental_dataset_path / "{}".format(snapshot) / "train2id.txt"
        write_to_file(file=pseud_incr_train2id_file, triples_iterable=pseud_incr_train2id_set)

        old_triple_result_set = new_triple_result_set


def store_triple_operations_to_incremental_folder(triple_operations_divided, incremental_dataset_path):
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


def calculate_and_store_snapshots(incremental_dataset_path, static_dataset_path, triple_operations_divided):
    triple_result_set = set()
    for snapshot_idx, triple_operations_list in enumerate(triple_operations_divided):
        for op in triple_operations_list:
            subj, objc, pred, op_type, ts = op
            triple = (subj, objc, pred)

            if op_type == "+":
                triple_result_set.add(triple)

            if op_type == "-":
                triple_result_set.remove(triple)

        snapshot = snapshot_idx + 1
        incremental_triple2id = incremental_dataset_path / "{}".format(snapshot) / "triple2id.txt"
        static_triple2id = static_dataset_path / "{}".format(snapshot) / "triple2id.txt"
        write_to_file(file=incremental_triple2id, triples_iterable=triple_result_set)
        write_to_file(file=static_triple2id, triples_iterable=triple_result_set)

def load_snapshot_triple_set(static_path, snapshot, filename="triple2id.txt"):
    triple_file = static_path / str(snapshot) / "{}".format(filename)
    triple_set = set()
    with triple_file.open(mode="rt", encoding="UTF-8") as f:
        for line in f:
            subjc, objc, pred = line.split()
            triple_set.add((subjc, objc, pred))
    return triple_set


def detect_added_and_deleted_triples(paths, prev_snaphot, curr_snapshot):
    static_path = paths["static_dataset_path"]
    old_triples_set = set() if prev_snaphot == 0 else load_snapshot_triple_set(static_path, prev_snaphot)
    new_triples_set = load_snapshot_triple_set(static_path, curr_snapshot)

    added_triples_set = new_triples_set - old_triples_set
    deleted_triples_set = old_triples_set - new_triples_set

    return added_triples_set, deleted_triples_set


def process_reinserts(added_triples_set, triple_sets, triple_histories):
    reinserted_triples_set = set()
    for triple in added_triples_set:
        # Reinserts for train triples
        if triple in triple_histories["train_triples_history"]:
            # Attach to train set to exclude it from split
            triple_sets["train_triples_set"].add(triple)
            reinserted_triples_set.add(triple)

        # Reinserts for valid triples
        if triple in triple_histories["valid_triples_history"]:
            # Attach to train set to exclude it from split
            triple_sets["valid_triples_set"].add(triple)
            reinserted_triples_set.add(triple)

        # Reinserts for test triples
        if triple in triple_histories["test_triples_history"]:
            # Attach to train set to exclude it from split
            triple_sets["test_triples_set"].add(triple)
            reinserted_triples_set.add(triple)

    return reinserted_triples_set


def perform_train_valid_test_split(added_triples_set):
    data = list(added_triples_set)
    # Split triple result list to 90% train and 10% eval triple
    train_triples, eval_triples = train_test_split(data, test_size=0.1, random_state=286)
    valid_triples, test_triples = train_test_split(eval_triples, test_size=0.5, random_state=286)

    return train_triples, valid_triples, test_triples


def assign_train_valid_test_data(triple_sets, triple_histories, train_triples, valid_triples, test_triples):
    triple_sets["train_triples_set"].update(train_triples)
    triple_histories["train_triples_history"].update(train_triples)

    triple_sets["valid_triples_set"].update(valid_triples)
    triple_histories["valid_triples_history"].update(valid_triples)

    triple_sets["test_triples_set"].update(test_triples)
    triple_histories["test_triples_history"].update(test_triples)


def remove_deleted_triples(deleted_triples_set, triple_sets):
    for triple in deleted_triples_set:
        if triple in triple_sets["train_triples_set"]:
            triple_sets["train_triples_set"].remove(triple)

        if triple in triple_sets["valid_triples_set"]:
            triple_sets["valid_triples_set"].remove(triple)

        if triple in triple_sets["test_triples_set"]:
            triple_sets["test_triples_set"].remove(triple)


def save_train_valid_test_sets(paths, triple_sets, snapshot):
    static_train_file = paths["static_dataset_path"] / "{}".format(snapshot) / "train2id.txt"
    write_to_file(file=static_train_file, triples_iterable=triple_sets["train_triples_set"])

    static_test_file = paths["static_dataset_path"] / "{}".format(snapshot) / "test2id.txt"
    write_to_file(file=static_test_file, triples_iterable=triple_sets["test_triples_set"])

    incr_test_file = paths["incremental_dataset_path"] / "{}".format(snapshot) / "test2id.txt"
    write_to_file(file=incr_test_file, triples_iterable=triple_sets["test_triples_set"])

    pseudo_incr_test_file = paths["pseudo_incremental_dataset_path"] / "{}".format(snapshot) / "test2id.txt"
    write_to_file(file=pseudo_incr_test_file, triples_iterable=triple_sets["test_triples_set"])

    static_valid_file = paths["static_dataset_path"] / "{}".format(snapshot) / "valid2id.txt"
    write_to_file(file=static_valid_file, triples_iterable=triple_sets["valid_triples_set"])

    incr_valid_file = paths["incremental_dataset_path"] / "{}".format(snapshot) / "valid2id.txt"
    write_to_file(file=incr_valid_file, triples_iterable=triple_sets["valid_triples_set"])

    pseudo_incr_valid_file = paths["pseudo_incremental_dataset_path"] / "{}".format(snapshot) / "valid2id.txt"
    write_to_file(file=pseudo_incr_valid_file, triples_iterable=triple_sets["valid_triples_set"])


def configure_train_valid_test_datasets(paths, num_snapshots):
    # (6) Generate train2id/ valid2id.txt/ test2id.txt for every snapshot. The training file will be used to learn
    # static models, while test and validation files are constructed for all types of methods.

    # Sets to track which tracks training, validation and test triples along the evolving KG
    triple_sets = {"train_triples_set": set(),
                   "valid_triples_set": set(),
                   "test_triples_set": set()}

    # Sets which track triples which have ever been inserted into sets
    triple_histories = {"train_triples_history": set(),
                        "valid_triples_history": set(),
                        "test_triples_history": set()}

    for snapshot in range(1, num_snapshots + 1):
        # (6.1) Process triple operations for each interval to determine which triples are added and deleted
        # Lists to track added and deleted triples per interval <snapshot_idx>
        added_triples_set, deleted_triples_set = detect_added_and_deleted_triples(paths, snapshot - 1, snapshot)

        # (6.2) Detect reinserts and attach them to corresponding set to ensure that triple do not jump between
        #         training, test and validation datasets
        reinserted_triple_set = process_reinserts(added_triples_set, triple_sets, triple_histories)
        added_triples_set = added_triples_set - reinserted_triple_set

        # (6.3) Train-/Test split on newly added triples and attach them to corresponding set
        train_triples, valid_triples, test_triples = perform_train_valid_test_split(added_triples_set)

        # (6.4) Update sets and histories
        assign_train_valid_test_data(triple_sets, triple_histories, train_triples, valid_triples, test_triples)

        # (6.5) Remove deleted triples with deletions occurring in interval <snapshot_idx+1> from these sets
        remove_deleted_triples(deleted_triples_set, triple_sets)

        # (5.1.4) Store Train-/Valid-/Test triples to files
        save_train_valid_test_sets(paths, triple_sets, snapshot)

def save_negative_triple_classification_files(deleted_triples_set, positive_oscillated_triples_set,
                                              negative_oscillated_triples_set, paths, dataset, snapshot):

    snapshot_folder = paths["incremental_dataset_path"] / "{}".format(snapshot)

    deleted_triple_file = snapshot_folder / "tc_negative_deleted_{}_triples.txt".format(dataset)
    positive_oscillated_file = snapshot_folder / "tc_positive_oscillated_{}_triples.txt".format(dataset)
    negative_oscillated_file = snapshot_folder / "tc_negative_oscillated_{}_triples.txt".format(dataset)

    if len(deleted_triples_set) > 0:
        write_to_tc_file(file=deleted_triple_file, triple_list=deleted_triples_set,
                         truth_value_list=[0] * len(deleted_triples_set))

    if len(negative_oscillated_triples_set) > 0:
        write_to_tc_file(file=negative_oscillated_file, triple_list=negative_oscillated_triples_set,
                         truth_value_list=[0] * len(negative_oscillated_triples_set))

    if len(positive_oscillated_triples_set) > 0:
        write_to_tc_file(file=positive_oscillated_file, triple_list=positive_oscillated_triples_set,
                         truth_value_list=[1] * len(positive_oscillated_triples_set))


def detect_and_store_deleted_and_oscillating_triples(paths, num_snapshots, dataset="train"):
    # (8) Detect reinserts and (positive | negative) oscillated triples for train, valid, test triples
    # Status Transitions:
    # (0)->(1) Inserted
    # (1)->(2) Deleted
    # (2)->(3) Positive Oscillated
    # (3)->(4) Negative Oscillated
    # (4)->(3) Positive Oscillated
    static_dataset_path = paths["static_dataset_path"]

    triples_status = {}
    deleted_triples_set = set()
    negative_oscillated_triples_set = set()
    positive_oscillated_triples_set = set()

    old_triple_set = set()
    for snapshot in range(1, num_snapshots + 1):
        # (8.1) Load train triples from snapshot <snapshot_idx>
        new_triple_set = load_snapshot_triple_set(static_dataset_path, snapshot, filename="{}2id.txt".format(dataset))

        # (8.2) Determine inserts and deletes
        inserts = new_triple_set - old_triple_set
        deletes = old_triple_set - new_triple_set

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

        # Switch old and new triple set for next snapshot
        old_triple_set = new_triple_set

        # (8.4) Store deleted and oscillated triples to files
        save_negative_triple_classification_files(deleted_triples_set, positive_oscillated_triples_set,
                                                  negative_oscillated_triples_set, paths, dataset, snapshot)

def configure_incremental_train_datasets(paths, triple_operations_divided, num_snapshots):
    new_train_triple_set = set()
    old_train_triple_set = set()

    static_dataset_path = paths["static_dataset_path"]
    for snapshot in range(1, num_snapshots + 1):
        # Load train triples from snapshot <snapshot_idx>
        new_train_triple_set = load_snapshot_triple_set(static_dataset_path, snapshot, "train2id.txt")

        # Determine inserts and deletes
        inserted_train_triples = new_train_triple_set - old_train_triple_set
        deleted_train_triples = old_train_triple_set - new_train_triple_set

        # Match inserted and deleted triples to their corresponding triple ops
        train_triple_operations = []
        triple_operations_list = triple_operations_divided[snapshot - 1]
        for operation in triple_operations_list:
            subj, objc, pred, op_type, ts = operation
            triple = (int(subj), int(objc), int(pred))

            if triple in inserted_train_triples and op_type == "+":
                train_triple_operations.append((subj, objc, pred, op_type, ts))
            if triple in deleted_train_triples and op_type == "-":
                train_triple_operations.append((subj, objc, pred, op_type, ts))

        # sort list of triple operations
        train_triple_operations = sort_triple_ops_list(train_triple_operations)

        # Save sorted train ops list
        incr_triple_op2id_file = paths["incremental_dataset_path"] / "{}".format(snapshot) / "train-op2id.txt"
        with incr_triple_op2id_file.open(mode="wt", encoding="UTF-8") as output:
            for triple_op in train_triple_operations:
                output_line = "{} {} {} {}\n".format(triple_op[0], triple_op[1], triple_op[2], triple_op[3])
                output.write(output_line)

        old_train_triple_set = new_train_triple_set


def create_wikidata_datasets(triple_operations, num_snapshots):
    output_path = Path.cwd() / "datasets_{}".format(datetime.now().strftime("%Y_%m_%d_%H_%M"))
    output_path.mkdir(exist_ok=True)

    # (1) Create directories and attach them to sub_dir_path
    paths_dict = create_directories(output_path, num_snapshots)

    # (2) Map item and property ids of wikidata to new global entity and relation ids which we use in our datasets
    triple_operations_mapped = create_global_mapping(triple_operations, output_path, paths_dict)

    # (3) Split (newly mapped) triple_operations into <num_snaps> parts and store them to triple-op2id.txt in the
    #     incremental dataset
    triple_operations_divided = divide_triple_operation_list(triple_operations_mapped, num_snapshots)

    # (4) Store triple operations for each interval to incremental dataset
    store_triple_operations_to_incremental_folder(triple_operations_divided, paths_dict["incremental_dataset_path"])

    # (5) Create set of present triple for each snapshot
    #     (Paths: [incremental | pseudo_incremental]/<snapshot>/triple2id.txt
    calculate_and_store_snapshots(paths_dict["incremental_dataset_path"], paths_dict["static_dataset_path"], triple_operations_divided)

    # (6) Generate train2id/ valid2id.txt/ test2id.txt for every snapshot. The training file will be used to learn
    # static models. Test and Validation files are constructed for all types of methods
    configure_train_valid_test_datasets(paths_dict, num_snapshots)

    # (6) Create triple2id.txt and train2id.txt for pseudo_incremental dataset
    create_pseudo_incremental_train_datasets(paths_dict, num_snapshots)

    # (7) Create train-op2id.txt
    configure_incremental_train_datasets(paths_dict, triple_operations_divided, num_snapshots)

    # (8) Create files for negative triple classification
    detect_and_store_deleted_and_oscillating_triples(paths_dict, num_snapshots, "train")
    detect_and_store_deleted_and_oscillating_triples(paths_dict, num_snapshots, "valid")
    detect_and_store_deleted_and_oscillating_triples(paths_dict, num_snapshots, "test")

def main():
    num_snapshots = 5
    triple_operation_file = Path.cwd() / "compiled_triple_operations" / "test_wikidata_kg_sorted_v2.txt.bz2"
    triple_operations = get_triple_operations(triple_operation_file)
    # triple_operations = get_compressed_triple_operations(triple_operation_file)
    create_wikidata_datasets(triple_operations, num_snapshots)


if __name__ == '__main__':
    main()