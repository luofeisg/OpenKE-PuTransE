# if false:
#     # Execute ParallelUniverse
#     from openke.data import TrainDataLoader, TestDataLoader
#
#     # dataloader for training
#     train_dataloader = TrainDataLoader(
#         in_path="./benchmarks/FB15K237/",
#         nbatches=100,
#         threads=8,
#         sampling_mode="normal",
#         bern_flag=1,
#         filter_flag=1,
#         neg_ent=25,
#         neg_rel=0)
#
#     import numpy as np
#
#     triple_constraint = 1000
#     balance_parameter = 0.5
#
#     entity_total = train_dataloader.lib.getEntityTotal()
#     entity_remapping = np.zeros(entity_total, dtype=np.int64)
#     relation_total = train_dataloader.lib.getRelationTotal()
#     relation_remapping = np.zeros(relation_total, dtype=np.int64)
#
#     h = np.zeros(triple_constraint, dtype=np.int64)
#     r = np.zeros(triple_constraint, dtype=np.int64)
#     t = np.zeros(triple_constraint, dtype=np.int64)
#
#     h_addr = h.__array_interface__["data"][0]
#     r_addr = r.__array_interface__["data"][0]
#     t_addr = t.__array_interface__["data"][0]
#     entity_remapping_addr = entity_remapping.__array_interface__["data"][0]
#     relation_remapping_addr = relation_remapping.__array_interface__["data"][0]
#
#     train_dataloader.lib.initializeSingleRandomSeed()
#     train_dataloader.lib.getParallelUniverse(
#         h_addr, r_addr, t_addr, entity_remapping_addr, relation_remapping_addr, triple_constraint, balance_parameter
#     )
#
#     # Examine number of duplicates
#     triples = []
#     for i in range(len(h)):
#         triples.append(tuple([h[i], r[i], t[i]]))
#
#     s = set(triples)
#     print(len(triples))
#     print(len(s))
#

if False:
    # Execute ParallelUniverse for all relations
    from openke.data import TrainDataLoader
    from openke.module.model import TransE

    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/FB15K237/",
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0)

    import numpy as np

    triple_constraint = 1000
    balance_parameter = 0.5
    relation_id = 100

    train_dataloader.lib.initializeSingleRandomSeed()
    train_dataloader.lib.getParallelUniverse(triple_constraint, balance_parameter, 10)

    relation_total_uni = train_dataloader.lib.getRelationTotalUniverse()
    relation_remapping = np.zeros(relation_total_uni)

    entity_total_uni = train_dataloader.lib.getEntityTotalUniverse()
    entity_remapping = np.zeros(entity_total_uni)

    next_universe_id = 0
    universe_dict = {}

    entity_occurences = dict(dict(set))  # length: train_dataloader.lib.getEntityTotal(), entity -> [Universes id]
    relation_occurences = dict(dict(set))  # length: train_dataloader.lib.getRelationTotal(), relation -> [Universes id]

    entity_mapping = dict(dict(int))  # universe_id -> entity_id -> index
    relation_mapping = dict(dict(int))  # universe_id -> relation_id -> index

    entity_remapping, relation_remapping = train_dataloader.get_universe_mappings()


    def process_universe_mapping(entity_remapping, relation_remapping):
        for index, entity in entity_remapping:
            entity_occurences[entity].add(next_universe_id)
            entity_mapping[next_universe_id][entity] = index
        for index, relation in relation_remapping:
            relation_occurences[relation].add(next_universe_id)
            relation_mapping[next_universe_id][relation] = index


# Execute ParallelUniverse for all relations
from openke.config import Trainer, Tester
from openke.module.model import TransE, ParallelUniverse
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

if False:
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/FB15K/",
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=2,
        neg_rel=0)


    PuTransE = ParallelUniverse(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        use_gpu=False,
        train_dataloader=train_dataloader,
        initial_num_universes=5000,
        min_margin=1,
        max_margin=4,
        min_lr=0.01,
        max_lr=0.1,
        min_num_epochs=50,
        max_num_epochs=200,
        min_triple_constraint=500,
        max_triple_constraint=2000,
        balance=0.5,
        num_dim=50,
        norm=2,
        embedding_meth="TransE")

    train_dataloader.lib.getParallelUniverse(1000, 0.5, 100)

    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/FB15K237/",
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0)

    # dataloader for test
    test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

    # define the model
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=200,
        p_norm=1,
        norm_flag=True)

    # define the loss function
    model = NegativeSampling(
        model=transe,
        loss=MarginLoss(margin=5.0),
        batch_size=train_dataloader.get_batch_size()
    )

    # train the model
    trainer = Trainer(model=model, data_loader=train_dataloader, train_times=1000, alpha=1.0, use_gpu=False)
    trainer.run()

########################################################################################
import openke

if __name__ == '__main__':
    from openke.module.model import ParallelUniverse
    from openke.data import TrainDataLoader, TestDataLoader

    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/FB15K/",
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=2,
        neg_rel=0)

    PuTransE = ParallelUniverse(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        use_gpu=False,
        train_dataloader=train_dataloader,
        initial_num_universes=5000,
        min_margin=1,
        max_margin=4,
        min_lr=0.01,
        max_lr=0.1,
        min_num_epochs=50,
        max_num_epochs=200,
        min_triple_constraint=500,
        max_triple_constraint=2000,
        balance=0.5,
        num_dim=50,
        norm=2,
        embedding_meth="TransE")

    PuTransE.train_embedding_space()
