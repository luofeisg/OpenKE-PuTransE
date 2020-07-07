from pathlib import Path
import shutil

def create_pseudo_incremental_updates(incremental_path):
    # incremental_path = Path.cwd() / "incremental"

    pseudo_incremental_path = Path.cwd() / "pseudo_incremental"
    pseudo_incremental_path.mkdir(exist_ok=True)

    # Copy entity2id.txt and relation2id.txt
    ent2id_file = incremental_path / "entity2id.txt"
    rel2id_file = incremental_path / "relation2id.txt"
    ps_ent2id_file = pseudo_incremental_path / "entity2id.txt"
    ps_rel2id_file = pseudo_incremental_path / "relation2id.txt"
    shutil.copy(str(ent2id_file), str(ps_ent2id_file))
    shutil.copy(str(rel2id_file), str(ps_rel2id_file))

    # Create pseudo incremental files
    incremental_update_folders = [fld for fld in incremental_path.iterdir() if fld.is_dir()]
    incremental_update_folders.sort()
    for update_folder in incremental_update_folders:
        test_file = update_folder / "test2id.txt"
        valid_file = update_folder / "valid2id.txt"

        # Create sub folder
        ps_subfolder = pseudo_incremental_path / update_folder.name
        ps_subfolder.mkdir(exist_ok=True)

        # Copy test file
        ps_test_file = ps_subfolder / test_file.name
        shutil.copy(str(test_file), str(ps_test_file))

        # Copy valid file
        ps_valid_file = ps_subfolder / valid_file.name
        shutil.copy(str(valid_file), str(ps_valid_file))

        # Create train set
        train_op_file = update_folder / "train-op2id.txt"

        # Get train result set
        train_base_set = set()
        with open(train_op_file, mode="rt", encoding="utf-8") as f:
            for line in f:
                head, tail, rel, op_type = line.split()
                triple = (int(head), int(tail), int(rel))
                if op_type == "+":
                    train_base_set.add(triple)
                elif op_type == "-":
                    if triple in train_base_set:
                        train_base_set.remove(triple)


        # Write train result set to file
        ps_train_file = ps_subfolder / "train2id.txt"
        with open(ps_train_file, mode="wt", encoding="utf-8") as out:
            for triple in train_base_set:
                head = triple[0]
                tail =triple[1]
                rel = triple[2]
                out.write("{} {} {}\n".format(head, tail, rel))

        # Create triple set
        triple_op_file = update_folder / "triple-op2id.txt"

        # Get triple result set
        triple_base_set = set()
        with open(triple_op_file, mode="rt", encoding="utf-8") as f:
            for line in f:
                head, tail, rel, op_type = line.split()
                triple = (int(head), int(tail), int(rel))
                if op_type == "+":
                    triple_base_set.add(triple)
                elif op_type == "-":
                    if triple in triple_base_set:
                        triple_base_set.remove(triple)

        # Write triple result set to file
        ps_triple_file = ps_subfolder / "triple2id.txt"
        with open(ps_triple_file, mode="wt", encoding="utf-8") as out:
            for triple in triple_base_set:
                head = triple[0]
                tail =triple[1]
                rel = triple[2]
                out.write("{} {} {}\n".format(head, tail, rel))

def create_static_snapshots(incremental_path):
    # incremental_path = Path.cwd() / "incremental"
    static_path = Path.cwd() / "static"
    static_path.mkdir(exist_ok=True)

    # Create pseudo incremental files
    incremental_update_folders = [fld for fld in incremental_path.iterdir() if fld.is_dir()]
    incremental_update_folders.sort()

    triple_base_set = set()
    train_base_set = set()
    for update_folder in incremental_update_folders:
        # Create sub folder
        static_subfolder = static_path / update_folder.name
        static_subfolder.mkdir(exist_ok=True)

        # Get train triple operations set from incremental update and calculate train result set
        train_op_file = update_folder / "train-op2id.txt"
        with open(train_op_file, mode="rt", encoding="utf-8") as f:
            for line in f:
                head, tail, rel, op_type = line.split()
                triple = (int(head), int(tail), int(rel))
                if op_type == "+":
                    train_base_set.add(triple)
                elif op_type == "-":
                    train_base_set.remove(triple)

        # Create entity2id.txt and relation2id.txt for train set
        next_ent_id = 0
        next_rel_id = 0
        entities_dict = {}
        relations_dict = {}

        # Write train result set to file with new entity and rel ids
        static_train_file = static_subfolder / "train2id.txt"
        with open(static_train_file, mode="wt", encoding="utf-8") as out:
            for triple in train_base_set:
                head = triple[0]
                tail = triple[1]
                rel = triple[2]

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

                out.write("{} {} {}\n".format(head, tail, rel))

        # Store entity2id and relation2id mapping
        static_ent2id_file = static_subfolder / "train_entity2id.txt"
        with open(static_ent2id_file, mode="wt", encoding="utf-8") as out:
            for global_entity_id, snap_entity_id in entities_dict.items():
                out.write("{} {}\n".format(snap_entity_id, global_entity_id))


        static_rel2id_file = static_subfolder / "train_relation2id.txt"
        with open(static_rel2id_file, mode="wt", encoding="utf-8") as out:
            for global_relation_id, snap_relation_id in relations_dict.items():
                out.write("{} {}\n".format(snap_relation_id, global_relation_id))


        ##############

        # Get triple operations set from incremental update and calculate triple2id result set
        triple_op_file = update_folder / "triple-op2id.txt"
        with open(triple_op_file, mode="rt", encoding="utf-8") as f:
            for line in f:
                head, tail, rel, op_type = line.split()
                triple = (int(head), int(tail), int(rel))
                if op_type == "+":
                    triple_base_set.add(triple)
                elif op_type == "-":
                    triple_base_set.remove(triple)

        # Create entity2id.txt and relation2id.txt for train set
        next_ent_id = 0
        next_rel_id = 0
        entities_dict = {}
        relations_dict = {}

        # Write train result set to file with new entity and rel ids
        static_triple_file = static_subfolder / "triple2id.txt"
        with open(static_triple_file, mode="wt", encoding="utf-8") as out:
            for triple in triple_base_set:
                head = triple[0]
                tail = triple[1]
                rel = triple[2]

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

                out.write("{} {} {}\n".format(head, tail, rel))

        # Store entity2id and relation2id mapping
        static_ent2id_file = static_subfolder / "entity2id.txt"
        with open(static_ent2id_file, mode="wt", encoding="utf-8") as out:
            for global_entity_id, snap_entity_id in entities_dict.items():
                out.write("{} {}\n".format(snap_entity_id, global_entity_id))

        static_rel2id_file = static_subfolder / "relation2id.txt"
        with open(static_rel2id_file, mode="wt", encoding="utf-8") as out:
            for global_relation_id, snap_relation_id in relations_dict.items():
                out.write("{} {}\n".format(snap_relation_id, global_relation_id))

        ############## TRAIN / VALID / TEST SPLIT (90 10 10)
        ############## TRAIN / VALID / TEST SPLIT (90 10 10)


        # Only for test purposes - filter out triples from eval files with unseen entities and relations
        # TEST
        test_file = update_folder / "test2id.txt"
        static_test_file = static_subfolder / "test2id.txt"
        seen_test_triples = []
        with open(test_file, mode="rt", encoding="utf-8") as f:
            for line in f:
                head, tail, rel = line.split()
                if head in entities_dict and tail in entities_dict and rel in relations_dict:
                    head = entities_dict[head]
                    tail = entities_dict[tail]
                    rel = relations_dict[rel]

                    seen_test_triples.append((head, tail, rel))

        with open(static_test_file, mode="wt", encoding="utf-8") as out:
            for triple in seen_test_triples:
                head, tail, rel = triple
                out.write("{} {} {}\n".format(head, tail, rel))

        # VALID
        valid_file = update_folder / "valid2id.txt"
        static_valid_file = static_subfolder / "valid2id.txt"
        seen_valid_triples = []
        with open(valid_file, mode="rt", encoding="utf-8") as f:
            for line in f:
                head, tail, rel = line.split()
                if head in entities_dict and tail in entities_dict and rel in relations_dict:
                    head = entities_dict[head]
                    tail = entities_dict[tail]
                    rel = relations_dict[rel]

                    seen_valid_triples.append((head, tail, rel))

        with open(static_valid_file, mode="wt", encoding="utf-8") as out:
            for triple in seen_valid_triples:
                head, tail, rel = triple
                out.write("{} {} {}\n".format(head, tail, rel))


def main():
    incremental_path = Path.cwd() / "incremental"
    # create_pseudo_incremental_updates(incremental_path)
    create_static_snapshots(incremental_path)

if __name__ == '__main__':
    main()