from openke.data import TestDataLoader
import numpy as np
import ctypes

class IncrementalTestDataLoader(TestDataLoader):
    def __init__(self, in_path="./benchmarks/Wikidata/datasets/incremental", sampling_mode='link', random_seed=4,
                 mode='test', setting="static", num_snapshots=None):
        super(IncrementalTestDataLoader, self).__init__(in_path=in_path, sampling_mode=sampling_mode,
                                                        random_seed=random_seed, mode=mode,
                                                        setting=setting)
        self.lib.initializeTripleOperations.argtypes = [ctypes.c_int64]
        self.lib.loadTestData.argtypes = [ctypes.c_int64]
        self.lib.loadValidData.argtypes = [ctypes.c_int64]

        self.num_snapshots = num_snapshots
        self.initialize_incremental_loading()

    def initialize_incremental_loading(self):
        # Constant variables along all snapshots
        self.lib.activateIncrementalSetting()
        # self.lib.readGlobalNumEntities()
        # self.lib.readGlobalNumRelations()
        self.entTotal = self.lib.getEntityTotal()
        self.relTotal = self.lib.getRelationTotal()
        self.lib.setNumSnapshots(self.num_snapshots)

    def evolveTripleList(self, snapshot_idx):
        self.lib.initializeTripleOperations(snapshot_idx)
        self.lib.evolveTripleList()

    def load_snapshot(self, snapshot_idx):
        if self.mode == "test":
            self.lib.loadTestData(snapshot_idx)
            self.testTotal = self.lib.getTestTotal()
        elif self.mode == "valid":
            self.lib.loadValidData(snapshot_idx)
            self.validTotal = self.lib.getValidTotal()

        # Number of currently contained entities in the KG at snapshot <snapshot_idx>
        # Derived from incrementally load triple operations along the snapshots
        self.currently_contained_entTotal = self.lib.getNumCurrentlyContainedEntities()
        self.update_batch_arrays()

    def update_batch_arrays(self):
        if self.mode == 'test':
            self.testTotal = self.lib.getTestTotal()

            self.test_h = np.zeros(self.currently_contained_entTotal, dtype=np.int64)
            self.test_t = np.zeros(self.currently_contained_entTotal, dtype=np.int64)
            self.test_r = np.zeros(self.currently_contained_entTotal, dtype=np.int64)
            self.test_h_addr = self.test_h.__array_interface__["data"][0]
            self.test_t_addr = self.test_t.__array_interface__["data"][0]
            self.test_r_addr = self.test_r.__array_interface__["data"][0]

            self.test_pos_h = np.zeros(self.testTotal, dtype=np.int64)
            self.test_pos_t = np.zeros(self.testTotal, dtype=np.int64)
            self.test_pos_r = np.zeros(self.testTotal, dtype=np.int64)
            self.test_pos_h_addr = self.test_pos_h.__array_interface__["data"][0]
            self.test_pos_t_addr = self.test_pos_t.__array_interface__["data"][0]
            self.test_pos_r_addr = self.test_pos_r.__array_interface__["data"][0]
            self.test_neg_h = np.zeros(self.testTotal, dtype=np.int64)
            self.test_neg_t = np.zeros(self.testTotal, dtype=np.int64)
            self.test_neg_r = np.zeros(self.testTotal, dtype=np.int64)
            self.test_neg_h_addr = self.test_neg_h.__array_interface__["data"][0]
            self.test_neg_t_addr = self.test_neg_t.__array_interface__["data"][0]
            self.test_neg_r_addr = self.test_neg_r.__array_interface__["data"][0]

        elif self.mode == 'valid':
            self.validTotal = self.lib.getValidTotal()

            self.valid_h = np.zeros(self.currently_contained_entTotal, dtype=np.int64)
            self.valid_t = np.zeros(self.currently_contained_entTotal, dtype=np.int64)
            self.valid_r = np.zeros(self.currently_contained_entTotal, dtype=np.int64)
            self.valid_h_addr = self.valid_h.__array_interface__["data"][0]
            self.valid_t_addr = self.valid_t.__array_interface__["data"][0]
            self.valid_r_addr = self.valid_r.__array_interface__["data"][0]

            self.valid_pos_h = np.zeros(self.validTotal, dtype=np.int64)
            self.valid_pos_t = np.zeros(self.validTotal, dtype=np.int64)
            self.valid_pos_r = np.zeros(self.validTotal, dtype=np.int64)
            self.valid_pos_h_addr = self.valid_pos_h.__array_interface__["data"][0]
            self.valid_pos_t_addr = self.valid_pos_t.__array_interface__["data"][0]
            self.valid_pos_r_addr = self.valid_pos_r.__array_interface__["data"][0]
            self.valid_neg_h = np.zeros(self.validTotal, dtype=np.int64)
            self.valid_neg_t = np.zeros(self.validTotal, dtype=np.int64)
            self.valid_neg_r = np.zeros(self.validTotal, dtype=np.int64)
            self.valid_neg_h_addr = self.valid_neg_h.__array_interface__["data"][0]
            self.valid_neg_t_addr = self.valid_neg_t.__array_interface__["data"][0]
            self.valid_neg_r_addr = self.valid_neg_r.__array_interface__["data"][0]
