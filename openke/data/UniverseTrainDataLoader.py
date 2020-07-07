# coding:utf-8
import os
import ctypes
import numpy as np
from .TrainDataLoader import TrainDataLoader


class UniverseTrainDataLoader(TrainDataLoader):

    def __init__(self, in_path="./", batch_size=None, nbatches=None, threads=8, sampling_mode="normal", bern_flag=0,
                 filter_flag=1, neg_ent=1, neg_rel=0, initial_random_seed=2):
        super(UniverseTrainDataLoader, self).__init__(in_path=in_path, batch_size=batch_size, nbatches=nbatches,
                                                      threads=threads, sampling_mode=sampling_mode, bern_flag=bern_flag,
                                                      filter_flag=filter_flag, neg_ent=neg_ent, neg_rel=neg_rel,
                                                      initial_random_seed=initial_random_seed)
        self.entity_total_universe = 0
        self.relation_total_universe = 0
        self.train_total_universe = 0

        """argtypes"""
        self.lib.sampling.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64
        ]

        self.lib.getParallelUniverse.argtypes = [
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_int64
        ]

        self.lib.getEntityRemapping.argtypes = [
            ctypes.c_void_p
        ]

        self.lib.getRelationRemapping.argtypes = [
            ctypes.c_void_p
        ]

        self.lib.getEntityTotalUniverse.restype = ctypes.c_int64
        self.lib.getRelationTotalUniverse.restype = ctypes.c_int64
        self.lib.getTrainTotalUniverse.restype = ctypes.c_int64

    def swap_helpers(self):
        self.lib.swapHelpers()

    def reset_universe(self):
        self.lib.resetUniverse()
        self.set_nbatches(self.lib.getTrainTotal, self.nbatches)

    def get_universe_mappings(self):
        entity_remapping = np.zeros(self.entity_total_universe, dtype=np.int64)
        relation_remapping = np.zeros(self.relation_total_universe, dtype=np.int64)

        entity_remapping_addr = entity_remapping.__array_interface__["data"][0]
        relation_remapping_addr = relation_remapping.__array_interface__["data"][0]

        self.lib.getEntityRemapping(entity_remapping_addr)
        self.lib.getRelationRemapping(relation_remapping_addr)
        return entity_remapping, relation_remapping

    def compile_universe_dataset(self, triple_constraint, balance_param, relation_in_focus):
        self.lib.getParallelUniverse(triple_constraint, balance_param, relation_in_focus)
        self.entity_total_universe = self.lib.getEntityTotalUniverse()
        self.relation_total_universe = self.lib.getRelationTotalUniverse()
        self.train_total_universe = self.lib.getTrainTotalUniverse()
        self.set_nbatches(self.train_total_universe, self.nbatches)
