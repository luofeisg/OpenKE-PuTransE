from openke.data import TrainDataLoader
from pathlib import Path

class IncrementalTrainDataLoader(TrainDataLoader):
    def __init__(self, in_path="./benchmarks/Wikidata/datasets/incremental", batch_size=None, nbatches=None, threads=8,
                 sampling_mode="normal", bern_flag=0,
                 filter_flag=1, neg_ent=1, neg_rel=0, random_seed=2, incremental_setting=False,
                 num_snapshots=None):
        super(IncrementalTrainDataLoader, self).__init__(in_path=in_path, batch_size=batch_size, nbatches=nbatches,
                                                         threads=threads, sampling_mode=sampling_mode,
                                                         bern_flag=bern_flag,
                                                         filter_flag=filter_flag, neg_ent=neg_ent, neg_rel=neg_rel,
                                                         random_seed=random_seed,
                                                         incremental_setting=incremental_setting)
        self.num_snapshots = num_snapshots
        self.initialize_incremental_loading()
        self.deleted_triple_set = set()

    def initialize_incremental_loading(self):
        # Constant variables along all snapshots
        self.lib.activateIncrementalSetting()
        self.lib.readGlobalNumEntities()
        self.lib.readGlobalNumRelations()
        self.relTotal = self.lib.getRelationTotal()
        self.entTotal = self.lib.getEntityTotal()
        self.lib.setNumSnapshots(self.num_snapshots)

    def load_snapshot(self, snapshot_idx):
        self.lib.initializeTrainingOperations(snapshot_idx)
        self.lib.evolveTrainList()

        # Update trainTotal and batchsize
        self.tripleTotal = self.lib.getTrainTotal()
        self.batch_size = self.tripleTotal // self.nbatches
        self.nbatches = self.tripleTotal // self.batch_size
        self.update_batch_arrays()

        self.track_deleted_triples(snapshot_idx)

    def track_deleted_triples(self, snapshot_idx):
        triple_operations_file = Path(self.in_path) / "incremental" / str(snapshot_idx) / "triple-op2id.txt"

        with open(triple_operations_file, mode="rt", encoding="utf-8") as f:
            for line in f:
                head, tail, rel, op_type = line.split()
                triple = (int(head), int(tail), int(rel))
                if op_type == "-":
                    self.deleted_triple_set.add(triple)
                elif op_type == "+":
                    if triple in self.deleted_triple_set:
                        self.deleted_triple_set.remove(triple)

