from openke.data import TrainDataLoader

class IncrementalTrainDataLoader(TrainDataLoader):
    def __init__(self, in_path="./benchmarks/Wikidata/datasets/incremental", batch_size=None, nbatches=None, threads=8,
                 sampling_mode="normal", bern_flag=0,
                 filter_flag=1, neg_ent=1, neg_rel=0, initial_random_seed=2, incremental_setting=False,
                 num_snapshots=None):
        super(IncrementalTrainDataLoader, self).__init__(in_path=in_path, batch_size=batch_size, nbatches=nbatches,
                                                         threads=threads, sampling_mode=sampling_mode,
                                                         bern_flag=bern_flag,
                                                         filter_flag=filter_flag, neg_ent=neg_ent, neg_rel=neg_rel,
                                                         initial_random_seed=initial_random_seed,
                                                         incremental_setting=incremental_setting)
        self.num_snapshots = num_snapshots
        self.initialize_incremental_loading()

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

        self.tripleTotal = self.lib.getTrainTotal()
        # update trainTotal and batchsize
        if self.batch_size == None:
            self.batch_size = self.tripleTotal // self.nbatches
        if self.nbatches == None:
            self.nbatches = self.tripleTotal // self.batch_size
        self.update_batch_arrays()

