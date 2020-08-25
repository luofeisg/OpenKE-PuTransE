from pathlib import Path
import bz2
import datetime
from tqdm import tqdm
from datetime import datetime
from collections import Counter, defaultdict
import shutil
from sklearn.model_selection import train_test_split
import random
import operator
from pprint import pprint
from random import randrange, sample


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
# |----------------------------entity2id.txt    (File) (see (6.6))
# |----------------------------relation2id.txt  (File) (see (6.6))
# |----------------------------triple2id.txt (triples die zu Snapshot1 noch wahr sind) (see (3.2.1))
# |----------------------------
# |----------------------------train2id.txt -\          | 90%
# |----------------------------valid2id.txt ---> Split  | 10% aus triple2id.txt  (see (6.7))
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


def create_global_mapping(triple_operations_divided, output_path, dataset_paths_dict):
    # (2) Map item and property ids of wikidata to new global entity and relation ids which we use in our datasets
    next_ent_id = 0
    next_rel_id = 0
    entities_dict = {}
    relations_dict = {}

    # (2.1) Iterate through triple operations and map wikidata_ids to new ids
    new_triple_operations_divided = []
    static_triple_file = output_path / "mapped_triple-op2id.txt"
    with static_triple_file.open(mode="wt", encoding="utf-8") as out:
        for snapshot_idx, triple_operations_list in enumerate(triple_operations_divided):
            triple_operations_mapped = []
            for triple_op in triple_operations_list:
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

                triple_operations_mapped.append((str(head), str(tail), str(rel), str(op_type), str(ts)))
                out.write("{} {} {} {} {}\n".format(head, tail, rel, op_type, ts))

            new_triple_operations_divided.append(triple_operations_mapped)

    # (2.2) Store entity2id and relation2id mapping
    print("Basic statistics: global number of entities: {}.".format(len(entities_dict)))
    global_ent2id_file = output_path / "entity2id.txt"
    with global_ent2id_file.open(mode="wt", encoding="utf-8") as out:
        for wikidata_id, mapped_id in entities_dict.items():
            out.write("{} {}\n".format(mapped_id, wikidata_id))

    print("Basic statistics: global number of relations: {}.".format(len(relations_dict)))
    global_rel2id_file = output_path / "relation2id.txt"
    with global_rel2id_file.open(mode="wt", encoding="utf-8") as out:
        for wikidata_id, mapped_id in relations_dict.items():
            out.write("{} {}\n".format(mapped_id, wikidata_id))

    # (2.3) Copy global entity2id and relation2id mapping to incremental and pseudo_incremental datasets
    incr_ent2id_file = dataset_paths_dict["incremental"] / "entity2id.txt"
    incr_rel2id_file = dataset_paths_dict["incremental"] / "relation2id.txt"
    shutil.copy(str(global_ent2id_file), str(incr_ent2id_file))
    shutil.copy(str(global_rel2id_file), str(incr_rel2id_file))

    ps_ent2id_file = dataset_paths_dict["pseudo_incremental"] / "entity2id.txt"
    ps_rel2id_file = dataset_paths_dict["pseudo_incremental"] / "relation2id.txt"
    shutil.copy(str(global_ent2id_file), str(ps_ent2id_file))
    shutil.copy(str(global_rel2id_file), str(ps_rel2id_file))

    return new_triple_operations_divided


def sort_triple_ops_list(triple_ops_list):
    sorted_triple_operations = sorted(triple_ops_list, key=operator.itemgetter(4, 0, 1, 2, 3))

    return sorted_triple_operations


def create_directories(output_path, num_snaps):
    subdir_names = ["static", "incremental", "pseudo_incremental"]
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
    static_dataset_path = paths["static"]
    pseudo_incremental_dataset_path = paths["pseudo_incremental"]

    old_triple_result_set = set()

    # (6.2) Iterate static snapshot folders and substract triple set in <n> from triple set in <n-1> into pseudo_incr/triple2id.txt
    for snapshot in range(1, num_snapshots + 1):
        new_triple_result_set = load_snapshot_triple_set(static_dataset_path, snapshot)

        # (6.3) Detect newly added triples in snapshot <n> by subtracting triple2id.txt from snapshot n - 1
        # and store into pseud_incr_dataset / <snapshot> / triple2id.txt
        pseud_incr_triple2id_set = new_triple_result_set - old_triple_result_set
        pseud_incr_triple2id_file = pseudo_incremental_dataset_path / "{}".format(snapshot) / "global_triple2id.txt"
        write_to_file(file=pseud_incr_triple2id_file, triples_iterable=pseud_incr_triple2id_set)

        # (6.4) Create train2id where only newly added train triples are covered
        # (6.4.1) Load train2id of snapshot
        new_train_triple_result_set = load_snapshot_triple_set(static_dataset_path, snapshot,
                                                               filename="train2id.txt")

        # (6.4.2) Detect newly added train triples in snapshot <n> by substracting train2id.txt from snap n with triple2id.txt of n - 1
        #         and store result into pseud_incr_dataset / <snapshot> / train2id.txt
        pseud_incr_train2id_set = new_train_triple_result_set - old_triple_result_set
        print("Snapshot {}: number of train triples (pseudo-incremental dataset): {}.".format(snapshot, len(
            pseud_incr_train2id_set)))
        pseud_incr_train2id_file = pseudo_incremental_dataset_path / "{}".format(snapshot) / "train2id.txt"
        write_to_file(file=pseud_incr_train2id_file, triples_iterable=pseud_incr_train2id_set)

        old_triple_result_set = new_triple_result_set


def store_triple_operations_to_incremental_folder(triple_operations_divided, incremental_dataset_path):
    for snapshot_idx, triple_operations_list in enumerate(triple_operations_divided):
        output_lines = []
        for op in triple_operations_list:
            subj, objc, pred, op_type, ts = op
            out_line = "{} {} {} {}\n".format(subj, objc, pred, op_type)
            output_lines.append(out_line)

        # Because we count from snapshot 1
        snapshot = snapshot_idx + 1
        output_file = incremental_dataset_path / "{}".format(snapshot) / "triple-op2id.txt"
        with output_file.open(mode="wt", encoding="UTF-8") as output:
            output.writelines(output_lines)


def calculate_and_store_snapshots(paths_dict, triple_operations_divided):
    incremental_dataset_path = paths_dict["incremental"]
    static_dataset_path = paths_dict["static"]

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
        incremental_triple2id = incremental_dataset_path / "{}".format(snapshot) / "global_triple2id.txt"
        static_triple2id = static_dataset_path / "{}".format(snapshot) / "global_triple2id.txt"
        write_to_file(file=incremental_triple2id, triples_iterable=triple_result_set)
        write_to_file(file=static_triple2id, triples_iterable=triple_result_set)



def load_snapshot_triple_set(path, snapshot, filename="global_triple2id.txt"):
    triple_file = path / str(snapshot) / "{}".format(filename)
    triple_set = set()
    with triple_file.open(mode="rt", encoding="UTF-8") as f:
        for line in f:
            subjc, objc, pred = line.split()
            triple_set.add((subjc, objc, pred))
    return triple_set


def detect_added_and_deleted_triples(paths, prev_snaphot, curr_snapshot):
    static_path = paths["static"]
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
    # Static train
    static_train_file = paths["static"] / "{}".format(snapshot) / "train2id.txt"
    write_to_file(file=static_train_file, triples_iterable=triple_sets["train_triples_set"])

    # Test (all)
    static_test_file = paths["static"] / "{}".format(snapshot) / "test2id.txt"
    write_to_file(file=static_test_file, triples_iterable=triple_sets["test_triples_set"])

    incr_test_file = paths["incremental"] / "{}".format(snapshot) / "test2id.txt"
    write_to_file(file=incr_test_file, triples_iterable=triple_sets["test_triples_set"])

    pseudo_incr_test_file = paths["pseudo_incremental"] / "{}".format(snapshot) / "test2id.txt"
    write_to_file(file=pseudo_incr_test_file, triples_iterable=triple_sets["test_triples_set"])

    # Valid (all)
    static_valid_file = paths["static"] / "{}".format(snapshot) / "valid2id.txt"
    write_to_file(file=static_valid_file, triples_iterable=triple_sets["valid_triples_set"])

    incr_valid_file = paths["incremental"] / "{}".format(snapshot) / "valid2id.txt"
    write_to_file(file=incr_valid_file, triples_iterable=triple_sets["valid_triples_set"])

    pseudo_incr_valid_file = paths["pseudo_incremental"] / "{}".format(snapshot) / "valid2id.txt"
    write_to_file(file=pseudo_incr_valid_file, triples_iterable=triple_sets["valid_triples_set"])

    print("Snapshot {}: number of validation triples: {}.".format(snapshot, len(triple_sets["valid_triples_set"])))
    print("Snapshot {}: number of test triples: {}.".format(snapshot, len(triple_sets["test_triples_set"])))


def create_entity_and_relation_mapping(paths, num_snapshots):
    for snapshot in range(1, num_snapshots + 1):
        create_snapshot_entity_and_relation_mapping(paths, snapshot)


def map_evaluation_samples(paths, snapshot, entities_dict, relations_dict):
    static_dataset_path = paths["static"]
    snapshot_fld = static_dataset_path / str(snapshot)
    dataset_filenames = ["valid2id.txt", "test2id.txt"]

    for dataset in dataset_filenames:
        triple_set = load_snapshot_triple_set(static_dataset_path, snapshot, dataset)
        output_file = snapshot_fld / dataset
        with output_file.open(mode="wt", encoding="UTF-8") as out:
            for triple in triple_set:
                head, tail, rel = triple
                head = entities_dict[head]
                tail = entities_dict[tail]
                rel = relations_dict[rel]

                # Write to local file
                out.write("{} {} {}\n".format(head, tail, rel))


def create_snapshot_entity_and_relation_mapping(paths, snapshot):
    static_dataset_path = paths["static"]
    snapshot_fld = static_dataset_path / str(snapshot)
    triple_set_filenames = ["train2id.txt", "valid2id.txt", "test2id.txt"]

    # (2) Map item and property ids of wikidata to new global entity and relation ids which we use in our datasets
    next_ent_id = 0
    next_rel_id = 0
    entities_dict = {}
    relations_dict = {}

    # (2.1) Iterate through triple operations and map wikidata_ids to new ids
    for dataset in triple_set_filenames:
        triple_set = load_snapshot_triple_set(static_dataset_path, snapshot, dataset)
        print("Snapshot {}: number of {} triples (static dataset): {}.".format(snapshot, dataset[:dataset.find("2")],
                                                                               len(triple_set)))

        static_triple_file = snapshot_fld / "{}".format(dataset)
        with static_triple_file.open(mode="wt", encoding="utf-8") as out:

            for triple in triple_set:
                head, tail, rel = triple

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

                # Write to local file
                out.write("{} {} {}\n".format(head, tail, rel))

    # (2.2) Store entity2id and relation2id mapping
    print("Snapshot {}: number of entities: {}.".format(snapshot, len(entities_dict)))
    ent2id_file = snapshot_fld / "entity2id.txt"
    with ent2id_file.open(mode="wt", encoding="utf-8") as out:
        for global_entity_id, local_snapshot_entity_id in entities_dict.items():
            out.write("{} {}\n".format(local_snapshot_entity_id, global_entity_id))

    print("Snapshot {}: number of relations: {}.".format(snapshot, len(relations_dict)))
    rel2id_file = snapshot_fld / "relation2id.txt"
    with rel2id_file.open(mode="wt", encoding="utf-8") as out:
        for global_relation_id, local_snapshot_relation_id in relations_dict.items():
            out.write("{} {}\n".format(local_snapshot_relation_id, global_relation_id))


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

        # (6.4) Update triple_sets and triple_histories
        assign_train_valid_test_data(triple_sets, triple_histories, train_triples, valid_triples, test_triples)

        # (6.5) Remove deleted triples with deletions occurring in interval <snapshot_idx+1> from these sets
        remove_deleted_triples(deleted_triples_set, triple_sets)

        # (6.6) Store Train-/Valid-/Test triples to files
        save_train_valid_test_sets(paths, triple_sets, snapshot)


def save_negative_triple_classification_files(deleted_triples_set, positive_oscillated_triples_set,
                                              negative_oscillated_triples_set, paths, dataset, snapshot):
    snapshot_folder = paths["incremental"] / "{}".format(snapshot)

    deleted_triple_file = snapshot_folder / "tc_negative_deleted_{}_triples.txt".format(dataset)
    positive_oscillated_file = snapshot_folder / "tc_positive_oscillated_{}_triples.txt".format(dataset)
    negative_oscillated_file = snapshot_folder / "tc_negative_oscillated_{}_triples.txt".format(dataset)

    print("Snapshot {}.".format(snapshot))
    print("-- Deleted {} triples: {}.".format(dataset, len(deleted_triples_set)))
    print("-- Positive oscillating {} triples: {}.".format(dataset, len(positive_oscillated_triples_set)))
    print("-- Negative oscillating {} triples: {}.\n".format(dataset, len(negative_oscillated_triples_set)))

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
    static_dataset_path = paths["static"]

    triples_status = {}
    deleted_triples_set = set()
    negative_oscillated_triples_set = set()
    positive_oscillated_triples_set = set()

    old_triple_set = set()
    for snapshot in range(1, num_snapshots + 1):
        # (8.1) Load train triples from snapshot <snapshot_idx>
        new_triple_set = load_snapshot_triple_set(static_dataset_path, snapshot,
                                                  filename="{}2id.txt".format(dataset))

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


def verify_consistency(triple_ops_list, snapshot):
    triple_ops_dict = defaultdict(list)
    triple_dict = defaultdict(int)
    for ops in triple_ops_list:
        subj, objc, pred, op_type, ts = ops
        triple = (subj, objc, pred)
        triple_dict[triple] = triple_dict[triple] + 1
        triple_ops_dict[triple].append((subj, objc, pred, op_type, ts))
        if triple_dict[triple] > 1:
            print("More than 1 operation for {} at snapshot {}.".format(triple, snapshot))
            pprint(triple_ops_dict[triple])


def configure_incremental_train_datasets(paths, triple_operations_divided, num_snapshots):
    old_train_triple_set = set()

    static_dataset_path = paths["static"]
    for snapshot in range(1, num_snapshots + 1):
        # Load train triples from snapshot <snapshot_idx>
        new_train_triple_set = load_snapshot_triple_set(static_dataset_path, snapshot, "train2id.txt")

        # Determine inserts and deletes
        inserted_train_triples = new_train_triple_set - old_train_triple_set
        deleted_train_triples = old_train_triple_set - new_train_triple_set

        # Match inserted and deleted triples to their corresponding triple ops
        train_triple_operations = []
        assigned_triples = set()
        triple_operations_list = triple_operations_divided[snapshot - 1]
        for operation in triple_operations_list:
            subj, objc, pred, op_type, ts = operation
            triple = (subj, objc, pred)

            if triple not in assigned_triples:
                if triple in inserted_train_triples and op_type == "+":
                    train_triple_operations.append((subj, objc, pred, op_type, ts))
                    assigned_triples.add(triple)
                if triple in deleted_train_triples and op_type == "-":
                    train_triple_operations.append((subj, objc, pred, op_type, ts))
                    assigned_triples.add(triple)

        # sort list of triple operations
        train_triple_operations = sort_triple_ops_list(train_triple_operations)
        verify_consistency(train_triple_operations, snapshot)
        # Save sorted train ops list
        print("Snapshot {}: number of train triple operations (incremental dataset): {}.".format(snapshot, len(
            train_triple_operations)))
        print("-- number of insert operations (incremental dataset): {}.".format(len(inserted_train_triples)))
        print("-- number of delete operations (incremental dataset): {}.".format(len(deleted_train_triples)))
        incr_triple_op2id_file = paths["incremental"] / "{}".format(snapshot) / "train-op2id.txt"
        with incr_triple_op2id_file.open(mode="wt", encoding="UTF-8") as output:
            for triple_op in train_triple_operations:
                output_line = "{} {} {} {}\n".format(triple_op[0], triple_op[1], triple_op[2], triple_op[3])
                output.write(output_line)

        old_train_triple_set = new_train_triple_set


def corrupt_triple(triple, mode, filter_set, entities_total):
    subj, objc, pred = triple
    while (True):
        corr_entity = randrange(0, entities_total)
        negative_triple = (corr_entity, objc, pred) if mode == "head" else (subj, corr_entity, pred)
        if negative_triple not in filter_set:
            break

    return negative_triple


def save_triple_classification_file(dataset_path, snapshot, positive_examples, negative_examples):
    output_file = dataset_path / str(snapshot) / "triple_classification_prepared_test_examples.txt"
    examples = positive_examples + negative_examples
    truthvalues = [1] * len(positive_examples) + [0] * len(negative_examples)
    write_to_tc_file(output_file, examples, truthvalues)


def create_incremental_triple_classification_file(paths_dict, num_snapshots):
    incremental_dataset_path = paths_dict["incremental"]
    global_entities_file = incremental_dataset_path / "entity2id.txt"
    entities_total = len(open(global_entities_file).readlines())
    print("-------------------------------------------------------------------------")
    print("Start gathering of persistent-negative examples for triple classification")
    print("Entities included in triple corruption process: {}.".format(entities_total))

    # Load all triples which have ever been inserted
    triple_set = set()
    for snapshot in range(1, num_snapshots + 1):
        snapshot_triples = load_snapshot_triple_set(incremental_dataset_path, snapshot, filename="global_triple2id.txt")
        triple_set.update(snapshot_triples)

    # Load all triples which have ever been inserted
    test_triple_set = set()
    for snapshot in range(1, num_snapshots + 1):
        snapshot_test_triples = load_snapshot_triple_set(incremental_dataset_path, snapshot,
                                                         filename="test2id.txt")
        test_triple_set.update(snapshot_test_triples)

    # Create pool of negative examples
    negative_triple_dict = {}  # test_triple -> negative example
    for triple in tqdm(test_triple_set):
        random_num = randrange(0, 10000)
        negative_triple = None

        # Corrupt head
        if random_num < 5000:
            negative_triple = corrupt_triple(triple, "head", triple_set, entities_total)

        # Corrupt tail
        elif random_num >= 5000:
            negative_triple = corrupt_triple(triple, "tail", triple_set, entities_total)

        negative_triple_dict[triple] = negative_triple

    # Iterate through snapshot to create triple_classification_file.txt from test2id.txt files by adding negative examples
    for snapshot in range(1, num_snapshots + 1):
        test_triples = list(load_snapshot_triple_set(incremental_dataset_path, snapshot, filename="test2id.txt"))

        # Gather negative examples
        negative_examples = []
        for triple in test_triples:
            negative_triple = negative_triple_dict[triple]
            negative_examples.append(negative_triple)

        save_triple_classification_file(incremental_dataset_path, snapshot, test_triples, negative_examples)


def sample_examples(paths_dict, snapshot, filename, num_samples):
    # Datasets to sample eval data for (Static datasets are created later)
    incremental_dataset_path = paths_dict["incremental"]
    pseudo_incremental_dataset_path = paths_dict["pseudo_incremental"]
    dataset_paths = [incremental_dataset_path, pseudo_incremental_dataset_path]

    triple_set = load_snapshot_triple_set(incremental_dataset_path, snapshot, filename)
    triple_set_sample = sample(triple_set, num_samples)

    for dataset_path in dataset_paths:
        # Rename file containing all triples to all_<filename>.txt (valid2id_all.txt | test2id_all.txt)
        snapshot_fld = dataset_path / "{}".format(snapshot)
        input_file = snapshot_fld / filename

        new_file_name = filename[:filename.find(".")] + "_all" + filename[filename.find("."):]
        input_file.rename(snapshot_fld / new_file_name)

        # Sample examples into new file with old name <filename>.txt (valid2id.txt | test2id.txt)
        sample_file = snapshot_fld / filename
        write_to_file(sample_file, triple_set_sample)


def sample_evaluation_examples(paths_dict, num_snapshots, num_samples):
    for snapshot in range(1, num_snapshots + 1):
        # sample test and valid data for all datasets
        sample_examples(paths_dict, snapshot, "valid2id.txt", num_samples)
        sample_examples(paths_dict, snapshot, "test2id.txt", num_samples)


def remove_obsolet_triple_ops(triple_operations_divided):
    removed_triples = []
    new_snapshots_triple_operations = []
    for snapshot_idx, triple_operations_list in enumerate(triple_operations_divided):
        # To track operations for a triple in the interval before a snapshot
        triple_operation_dict = defaultdict(list)
        for triple_op in triple_operations_list:
            subj, objc, pred, op_type, ts = triple_op

            if subj == objc:
                continue

            triple = (subj, objc, pred)
            triple_operation_dict[triple].append(triple_op)

        # Determine all triples with a odd number of triple operations because
        # those with even numbers are inserted and deleted within the same interval
        filtered_triple_operations = []
        for triple, triple_op_list in triple_operation_dict.items():
            if len(triple_op_list) % 2 != 0:
                filtered_triple_operations.append(triple_op_list[-1])
            else:
                removed_triples.append(triple)

        sorted(filtered_triple_operations, key=operator.itemgetter(4, 0, 1, 2, 3))
        new_snapshots_triple_operations.append(filtered_triple_operations)

    return new_snapshots_triple_operations


def remove_uncommon_triple_ops(triple_operations_divided, num_snapshots, entity_frequencies_threshold,
                               relation_frequencies_threshold):
    entity_occ_counter = defaultdict(lambda: {i: 0 for i in range(1, num_snapshots + 1)})
    relation_occ_counter = defaultdict(lambda: {i: 0 for i in range(1, num_snapshots + 1)})

    triple_result_set = set()
    for snapshot_idx, triple_operations_list in enumerate(triple_operations_divided):
        for triple_op in triple_operations_list:
            subj, objc, pred, op_type, ts = triple_op
            triple = (subj, objc, pred)
            if op_type == "+":
                triple_result_set.add(triple)
            elif op_type == "-":
                triple_result_set.remove(triple)

        # Iterate through triple result list to obtain frequencies
        for triple in triple_result_set:
            subj, objc, pred = triple
            entity_occ_counter[subj][snapshot_idx + 1] += 1
            entity_occ_counter[objc][snapshot_idx + 1] += 1
            relation_occ_counter[pred][snapshot_idx + 1] += 1

    uncommon_relations = set()
    for relation, snapshot_frequencies_dict in relation_occ_counter.items():
        for snapshot, count in snapshot_frequencies_dict.items():
            if count > 0 and count < relation_frequencies_threshold:
                uncommon_relations.add(relation)
                break

    uncommon_entities = set()
    for entity, snapshot_frequencies_dict in entity_occ_counter.items():
        for snapshot, count in snapshot_frequencies_dict.items():
            if count < entity_frequencies_threshold and count > 0:
                uncommon_entities.add(entity)
                break

    filtered_triple_operations_divided = []
    for snapshot_idx, triple_operations_list in enumerate(triple_operations_divided):
        filtered_ops = []
        for triple_op in triple_operations_list:
            subj, objc, pred, op_type, ts = triple_op

            if subj not in uncommon_entities \
                    and objc not in uncommon_entities \
                    and pred not in uncommon_relations:
                filtered_ops.append(triple_op)

        filtered_triple_operations_divided.append(filtered_ops)

    return filtered_triple_operations_divided

def load_mapping_dict(filepath):
    mapping_dict = {}
    with filepath.open(mode="rt", encoding="UTF-8") as f:
        for line in f:
            local_id, global_id = line.split()
            mapping_dict[global_id] = local_id

    return mapping_dict

def map_triple_files(path_dict, num_snapshots):
    static_dataset_path = path_dict["static"]
    dataset_name = "global_triple2id.txt"

    for snapshot in range(1, num_snapshots):
        static_snapshot_fld = static_dataset_path / "{}".format(snapshot)
        entities_mapping_file =  static_snapshot_fld / "entity2id.txt"
        entity_mapping_dict = load_mapping_dict(entities_mapping_file)
        relations_mapping_file = static_snapshot_fld / "relation2id.txt"
        relation_mapping_dict = load_mapping_dict(relations_mapping_file)

        output_file_name = dataset_name[dataset_name.find("triple"):]
        output_file = static_snapshot_fld / output_file_name
        input_triple_set = load_snapshot_triple_set(static_dataset_path, snapshot, dataset_name)

        with output_file.open(mode="wt", encoding="UTF-8") as out:
            for triple in input_triple_set:
                head, tail, rel = triple
                head = entity_mapping_dict[head]
                tail = entity_mapping_dict[tail]
                rel = relation_mapping_dict[rel]
                out.write("{} {} {}\n".format(head, tail, rel))

def map_static_evaluation_samples(path_dict, num_snapshots):
    incremental_dataset_path = path_dict["incremental"]
    static_dataset_path = path_dict["static"]
    dataset_names = ["valid2id.txt", "test2id.txt"]

    for snapshot in range(1, num_snapshots):
        static_snapshot_fld = static_dataset_path / "{}".format(snapshot)
        entities_mapping_file =  static_snapshot_fld / "entity2id.txt"
        entity_mapping_dict = load_mapping_dict(entities_mapping_file)

        relations_mapping_file =  static_snapshot_fld / "relation2id.txt"
        relation_mapping_dict = load_mapping_dict(relations_mapping_file)

        for dataset in dataset_names:
            output_file_name = "sample_{}".format(dataset)
            output_file = static_snapshot_fld / output_file_name
            input_triple_set = load_snapshot_triple_set(incremental_dataset_path, snapshot, dataset)
            with output_file.open(mode="wt", encoding="UTF-8") as out:
                for triple in input_triple_set:
                    head, tail, rel = triple
                    head = entity_mapping_dict[head]
                    tail = entity_mapping_dict[tail]
                    rel = relation_mapping_dict[rel]
                    out.write("{} {} {}\n".format(head, tail, rel))

            # Switch sample and file with all valid | test examples
            all_triples_file = static_snapshot_fld / dataset
            new_file_name = dataset[:dataset.find(".")] + "_all" + dataset[dataset.find("."):]

            all_triples_file.rename(static_snapshot_fld / new_file_name)
            output_file.rename(static_snapshot_fld / dataset)


def create_wikidata_datasets(triple_operations, num_snapshots, num_of_sampled_test_triples=None):
    print("Begin compilation of experimental datasets at {}.".format(datetime.now().strftime("%H:%M:%S")))

    random_seed = 2861992
    print("Set random seed: {}".format(random_seed))
    random.seed(random_seed)

    output_path = Path.cwd() / "WikidataEvolve"
    # output_path = Path.cwd() / "datasets_{}".format(datetime.now().strftime("%Y_%m_%d_%H_%M"))
    output_path.mkdir(exist_ok=True)

    # (1) Create directories and attach them to sub_dir_path
    paths_dict = create_directories(output_path, num_snapshots)

    # (2) Split (newly mapped) triple_operations into <num_snaps> parts
    triple_operations_divided = divide_triple_operation_list(triple_operations, num_snapshots)

    # (3) FILTERING
    # (3.1) Remove obsolete filter operations (e.g. Insert and Delete Operation of same triple in same interval)
    triple_operations_divided = remove_obsolet_triple_ops(triple_operations_divided)

    # (3.2) Remove filter operations of entities and relation which participate in less than 50 triples along the kg
    triple_operations_divided = remove_uncommon_triple_ops(triple_operations_divided,
                                                           num_snapshots,
                                                           entity_frequencies_threshold=15,
                                                           relation_frequencies_threshold=100)

    # (3) Map item and property ids of wikidata to new global entity and relation ids which we use in our datasets
    triple_operations_divided = create_global_mapping(triple_operations_divided, output_path, paths_dict)

    # (4) Store triple operations for each interval to incremental dataset
    store_triple_operations_to_incremental_folder(triple_operations_divided, paths_dict["incremental"])

    # (5) Create set of present triple for each snapshot
    #     (Paths: [incremental | pseudo_incremental]/<snapshot>/global_triple2id.txt
    calculate_and_store_snapshots(paths_dict, triple_operations_divided)

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

    # (9) Sample valid/ test examples to make evaluation more efficient
    if num_of_sampled_test_triples:
        sample_evaluation_examples(paths_dict, num_snapshots, num_of_sampled_test_triples)

    # (10) Map datasets to local_ids at each snapshot for static learning procedure
    create_entity_and_relation_mapping(paths_dict, num_snapshots)

    if num_of_sampled_test_triples:
        map_static_evaluation_samples(paths_dict, num_snapshots)
    map_triple_files(paths_dict, num_snapshots)

    # (11)
    create_incremental_triple_classification_file(paths_dict, num_snapshots)
    print("Finished compilation process at {}.".format(datetime.now().strftime("%H:%M:%S")))


def main():
    num_snapshots = 4
    triple_operation_file = Path.cwd() / "compiled_triple_operations" / "compiled_triple_operations_directly_filtered_and_sorted.txt.bz2"
    triple_operations = get_triple_operations(triple_operation_file)
    # split in half to reduce amount of data
    # triple_operations = triple_operations[:len(triple_operations)//3]
    create_wikidata_datasets(triple_operations, num_snapshots, num_of_sampled_test_triples=6000)


if __name__ == '__main__':
    main()
