# coding:utf-8
import os
import ctypes
import numpy as np


class TestDataSampler(object):

    def __init__(self, data_total, data_sampler):
        self.data_total = data_total
        self.data_sampler = data_sampler
        self.total = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.total += 1
        if self.total > self.data_total:
            raise StopIteration()
        return self.data_sampler()

    def __len__(self):
        return self.data_total


class TestDataLoader(object):

    def __init__(self, in_path="./", sampling_mode='link', random_seed=4, mode='test', setting="static"):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        # print("Random_seed for TestDataLoader: {}".format(self.lib.getRandomSeed()))
        self.setting = setting
        self.mode = mode
        if self.mode == 'test':
            """for link prediction"""
            self.lib.getHeadBatch.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            self.lib.getTailBatch.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            """for triple classification"""
            self.lib.getTestBatch.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
        elif self.mode == 'valid':
            """'valid"""
            self.lib.getValidHeadBatch.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            self.lib.getValidTailBatch.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]

            self.lib.validHead.argtypes = [ctypes.c_void_p]
            self.lib.validTail.argtypes = [ctypes.c_void_p]

        self.lib.setRandomSeed.argtypes = [
            ctypes.c_int64
        ]
        """set essential parameters"""
        self.in_path = in_path
        self.sampling_mode = sampling_mode
        self.random_seed = random_seed
        self.read()

    def read(self):
        self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
        self.lib.setRandomSeed(self.random_seed)
        self.lib.randReset()

        if self.setting == "static":
            # delegated importTrainFiles execution to python because corruption process in sampling for triple classification
            # accesses training data structures in order to filter negative examples that did not occurred in train data
            self.lib.importTrainFiles() # Only necessary if we evaluate without creating a TrainDataLoader object before
            self.lib.importTestFiles()
            self.relTotal = self.lib.getRelationTotal()
            self.entTotal = self.lib.getEntityTotal()

            if self.mode == 'test':
                self.testTotal = self.lib.getTestTotal()

                self.test_h = np.zeros(self.entTotal, dtype=np.int64)
                self.test_t = np.zeros(self.entTotal, dtype=np.int64)
                self.test_r = np.zeros(self.entTotal, dtype=np.int64)
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

                self.valid_h = np.zeros(self.entTotal, dtype=np.int64)
                self.valid_t = np.zeros(self.entTotal, dtype=np.int64)
                self.valid_r = np.zeros(self.entTotal, dtype=np.int64)
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

    def sampling_lp(self):
        res = []
        if self.mode == 'test':
            self.lib.getHeadBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
            res.append({
                "batch_h": self.test_h.copy(),
                "batch_t": self.test_t[:1].copy(),
                "batch_r": self.test_r[:1].copy(),
                "mode": "head_batch"
            })
            self.lib.getTailBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
            res.append({
                "batch_h": self.test_h[:1].copy(),
                "batch_t": self.test_t.copy(),
                "batch_r": self.test_r[:1].copy(),
                "mode": "tail_batch"
            })
        elif self.mode == 'valid':
            self.lib.getValidHeadBatch(self.valid_h_addr, self.valid_t_addr, self.valid_r_addr)
            res.append({
                "batch_h": self.valid_h.copy(),
                "batch_t": self.valid_t[:1].copy(),
                "batch_r": self.valid_r[:1].copy(),
                "mode": "head_batch"
            })
            self.lib.getValidTailBatch(self.valid_h_addr, self.valid_t_addr, self.valid_r_addr)
            res.append({
                "batch_h": self.valid_h[:1].copy(),
                "batch_t": self.valid_t.copy(),
                "batch_r": self.valid_r[:1].copy(),
                "mode": "tail_batch"
            })

        return res

    def sampling_tc(self):
        self.lib.getTestBatch(
            self.test_pos_h_addr,
            self.test_pos_t_addr,
            self.test_pos_r_addr,
            self.test_neg_h_addr,
            self.test_neg_t_addr,
            self.test_neg_r_addr,
        )
        return [
            {
                'batch_h': self.test_pos_h,
                'batch_t': self.test_pos_t,
                'batch_r': self.test_pos_r,
                "mode": "normal"
            },
            {
                'batch_h': self.test_neg_h,
                'batch_t': self.test_neg_t,
                'batch_r': self.test_neg_r,
                "mode": "normal"
            }
        ]

    """interfaces to get essential parameters"""

    def get_ent_tot(self):
        return self.entTotal

    def get_rel_tot(self):
        return self.relTotal

    def get_triple_tot(self):
        return self.testTotal

    def set_sampling_mode(self, sampling_mode):
        self.sampling_mode = sampling_mode

    def __len__(self):
        return self.testTotal if self.mode == 'test' else self.validTotal

    def __iter__(self):
        if self.sampling_mode == 'link':
            if self.mode == 'test':
                self.lib.initTest()
                eval_total = self.testTotal
            elif self.mode == 'valid':
                self.lib.validInit()
                eval_total = self.validTotal

            return TestDataSampler(eval_total, self.sampling_lp)
        else:
            self.lib.initTest()
            return TestDataSampler(1, self.sampling_tc)
