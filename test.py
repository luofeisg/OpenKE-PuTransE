#Execute ParallelUniverse
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/FB15K237/",
	nbatches = 100,
	threads = 8,
	sampling_mode = "normal",
	bern_flag = 1,
	filter_flag = 1,
	neg_ent = 25,
	neg_rel = 0)


import numpy as np

triple_constraint = 1000
balance_parameter = 0.5

h = np.zeros(triple_constraint, dtype=np.int64)
t = np.zeros(triple_constraint, dtype=np.int64)
r = np.zeros(triple_constraint, dtype=np.int64)


h_addr = h.__array_interface__["data"][0]
r_addr = r.__array_interface__["data"][0]
t_addr = t.__array_interface__["data"][0]

train_dataloader.lib.getParallelUniverse(
    h_addr,t_addr,r_addr,triple_constraint,balance_parameter
)


# Examine numer of duplicates
triples = []
for i in range(len(h)):
    triples.append(tuple([h[i],r[i],t[i]]))

s = set(triples)
print(len(triples))
print(len(s))