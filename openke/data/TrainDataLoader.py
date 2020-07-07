# coding:utf-8
import os
import ctypes
import numpy as np


class TrainDataSampler(object):

    def __init__(self, nbatches, datasampler):
        self.nbatches = nbatches
        self.datasampler = datasampler
        self.batch = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.batch += 1
        if self.batch > self.nbatches:
            raise StopIteration()
        return self.datasampler()

    def __len__(self):
        return self.nbatches


class TrainDataLoader(object):
    def __init__(self, in_path="./", batch_size=None, nbatches=None, threads=8, sampling_mode="normal", bern_flag=0,
                 filter_flag=1, neg_ent=1, neg_rel=0, random_seed=2, incremental_setting=False):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        """argtypes"""
        self.lib.sampling.argtypes = [
            ctypes.c_void_p,
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
            ctypes.c_float
        ]

        self.lib.getEntityRemapping.argtypes = [
            ctypes.c_void_p
        ]

        self.lib.getRelationRemapping.argtypes = [
            ctypes.c_void_p
        ]

        self.lib.getNumOfNegatives.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64
        ]

        self.lib.getNumOfNegatives.restype = ctypes.c_int64

        self.lib.getNegativeEntities.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64
        ]

        self.lib.getNumOfPositives.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64
        ]

        self.lib.getNumOfPositives.restype = ctypes.c_int64

        self.lib.getPositiveEntities.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64
        ]

        self.lib.getNumOfEntityRelations.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64
        ]

        self.lib.getNumOfEntityRelations.restype = ctypes.c_int64

        self.lib.getEntityRelations.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64
        ]

        """set essential parameters"""
        self.in_path = in_path
        self.work_threads = threads
        self.nbatches = nbatches
        self.batch_size = batch_size
        self.bern = bern_flag
        self.filter = filter_flag
        self.negative_ent = neg_ent
        self.negative_rel = neg_rel
        self.sampling_mode = sampling_mode
        self.cross_sampling_flag = 0
        self.random_seed = random_seed
        self.incremental_setting = incremental_setting
        self.read()

    def read(self):
        self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
        self.lib.setBern(self.bern)
        self.lib.setWorkThreads(self.work_threads)
        self.lib.setRandomSeed(self.random_seed)
        self.lib.randReset()

        if not self.incremental_setting:
            self.lib.importTrainFiles()
            self.relTotal = self.lib.getRelationTotal()
            self.entTotal = self.lib.getEntityTotal()
            self.tripleTotal = self.lib.getTrainTotal()


            if self.batch_size == None:
                self.batch_size = self.tripleTotal // self.nbatches
            if self.nbatches == None:
                self.nbatches = self.tripleTotal // self.batch_size
            self.update_batch_arrays()

    def update_batch_arrays(self):
        self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)

        self.batch_h = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_t = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_r = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_y = np.zeros(self.batch_seq_size, dtype=np.float32)
        self.batch_head_corr = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_h_addr = self.batch_h.__array_interface__["data"][0]
        self.batch_t_addr = self.batch_t.__array_interface__["data"][0]
        self.batch_r_addr = self.batch_r.__array_interface__["data"][0]
        self.batch_y_addr = self.batch_y.__array_interface__["data"][0]
        self.batch_head_corr_addr = self.batch_head_corr.__array_interface__["data"][0]

    def swap_helpers(self):
        self.lib.swapHelpers()

    def reset_universe(self):
        self.lib.resetUniverse()
        self.set_nbatches(self.lib.getTrainTotal(), self.nbatches)

    def get_universe_mappings(self):
        entity_total_universe = self.lib.getEntityTotalUniverse()
        relation_total_universe = self.lib.getRelationTotalUniverse()

        entity_remapping = np.zeros(entity_total_universe, dtype=np.int64)
        relation_remapping = np.zeros(relation_total_universe, dtype=np.int64)

        entity_remapping_addr = entity_remapping.__array_interface__["data"][0]
        relation_remapping_addr = relation_remapping.__array_interface__["data"][0]

        self.lib.getEntityRemapping(entity_remapping_addr)
        self.lib.getRelationRemapping(relation_remapping_addr)
        return entity_remapping, relation_remapping

    def compile_universe_dataset(self, triple_constraint, balance_param):
        self.lib.getParallelUniverse(triple_constraint, balance_param)
        self.set_nbatches(self.lib.getTrainTotalUniverse(), self.nbatches)

    def sampling(self):
        self.lib.sampling(
            self.batch_h_addr,
            self.batch_t_addr,
            self.batch_r_addr,
            self.batch_y_addr,
            self.batch_head_corr_addr,
            self.batch_size,
            self.negative_ent,
            self.negative_rel,
            0,
            self.filter,
            0,
            0
        )
        return {
            "batch_h": self.batch_h,
            "batch_t": self.batch_t,
            "batch_r": self.batch_r,
            "batch_y": self.batch_y,
            "batch_head_corr": self.batch_head_corr,
            "mode": "normal"
        }

    def sampling_head(self):
        self.lib.sampling(
            self.batch_h_addr,
            self.batch_t_addr,
            self.batch_r_addr,
            self.batch_y_addr,
            self.batch_head_corr_addr,
            self.batch_size,
            self.negative_ent,
            self.negative_rel,
            -1,
            self.filter,
            0,
            0
        )
        return {
            "batch_h": self.batch_h,
            "batch_t": self.batch_t[:self.batch_size],
            "batch_r": self.batch_r[:self.batch_size],
            "batch_y": self.batch_y,
            "batch_head_corr": self.batch_head_corr,
            "mode": "head_batch"
        }

    def sampling_tail(self):
        self.lib.sampling(
            self.batch_h_addr,
            self.batch_t_addr,
            self.batch_r_addr,
            self.batch_y_addr,
            self.batch_head_corr_addr,
            self.batch_size,
            self.negative_ent,
            self.negative_rel,
            1,
            self.filter,
            0,
            0
        )
        return {
            "batch_h": self.batch_h[:self.batch_size],
            "batch_t": self.batch_t,
            "batch_r": self.batch_r[:self.batch_size],
            "batch_y": self.batch_y,
            "batch_head_corr": self.batch_head_corr,
            "mode": "tail_batch"
        }

    def cross_sampling(self):
        self.cross_sampling_flag = 1 - self.cross_sampling_flag
        # self.cross_sampling_flag = 0 #haha
        if self.cross_sampling_flag == 0:
            return self.sampling_head()
        else:
            return self.sampling_tail()

    def get_positive_entities(self, entity, relation, entity_is_head):
        num_of_pos = self.lib.getNumOfPositives(entity, relation, entity_is_head)
        batch_pos_entities = np.zeros(num_of_pos, dtype=np.int64)
        batch_pos_entities_addr = batch_pos_entities.__array_interface__["data"][0]
        self.lib.getPositiveEntities(batch_pos_entities_addr, entity, relation, entity_is_head)

        return batch_pos_entities

    def get_negative_entities(self, entity, relation, entity_is_head):
        num_of_neg = self.lib.getNumOfNegatives(entity, relation, entity_is_head)
        batch_neg_entities = np.zeros(num_of_neg, dtype=np.int64)

        if num_of_neg != 0:
            batch_neg_entities_addr = batch_neg_entities.__array_interface__["data"][0]
            self.lib.getNegativeEntities(batch_neg_entities_addr, entity, relation, entity_is_head)

        return batch_neg_entities

    def get_entity_relations(self, entity, entity_is_tail):
        num_of_rel = self.lib.getNumOfEntityRelations(entity, entity_is_tail)
        batch_entity_relations = np.zeros(num_of_rel, dtype=np.int64)

        batch_entity_relations_addr = batch_entity_relations.__array_interface__["data"][0]
        self.lib.getEntityRelations(batch_entity_relations_addr, entity, entity_is_tail)

        return batch_entity_relations

    """interfaces to set essential parameters"""

    def set_work_threads(self, work_threads):
        self.work_threads = work_threads

    def set_in_path(self, in_path):
        self.in_path = in_path

    def set_nbatches(self, triple_total, nbatches):
        self.nbatches = nbatches
        self.batch_size = triple_total // nbatches
        self.update_batch_arrays()

    def set_batch_size(self, triple_total, batch_size):
        self.nbatches = triple_total // batch_size
        self.batch_size = batch_size
        self.update_batch_arrays()

    def set_ent_neg_rate(self, rate):
        self.negative_ent = rate

    def set_rel_neg_rate(self, rate):
        self.negative_rel = rate

    def set_bern_flag(self, bern):
        self.bern = bern

    def set_filter_flag(self, filter):
        self.filter = filter

    """interfaces to get essential parameters"""

    def get_batch_size(self):
        return self.batch_size

    def get_ent_tot(self):
        return self.entTotal

    def get_rel_tot(self):
        return self.relTotal

    def get_triple_tot(self):
        return self.tripleTotal

    def __iter__(self):
        if self.sampling_mode == "normal":
            return TrainDataSampler(self.nbatches, self.sampling)
        else:
            return TrainDataSampler(self.nbatches, self.cross_sampling)

    def __len__(self):
        return self.nbatches
