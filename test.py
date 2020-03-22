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

# if __name__ == '__main__':
#     from openke.module.model import ParallelUniverse
#     from openke.data import TrainDataLoader, TestDataLoader
#     from openke.config import Trainer, Tester
#     from collections import defaultdict
#     import torch
#
#     train_dataloader = TrainDataLoader(
#         in_path="./benchmarks/WN18/",
#         nbatches=100,
#         threads=8,
#         sampling_mode="normal",
#         bern_flag=1,
#         filter_flag=0,
#         neg_ent=1,
#         neg_rel=0)
#
#     PuTransE = ParallelUniverse(
#         ent_tot=train_dataloader.get_ent_tot(),
#         rel_tot=train_dataloader.get_rel_tot(),
#         train_dataloader=train_dataloader,
#         initial_num_universes=5000,
#         min_margin=1,
#         max_margin=4,
#         min_lr=0.01,
#         max_lr=0.1,
#         min_num_epochs=50,
#         max_num_epochs=200,
#         min_triple_constraint=500,
#         max_triple_constraint=2000,
#         balance=0.5,
#         num_dim=50,
#         norm=1,
#         embedding_method="TransE",
#         checkpoint_dir="./checkpoint/",
#         save_steps=2)
#
#     # PuTransE.train_embedding_space()
#
#     state_dict = torch.load("./checkpoint/PuTransE_learned_spaces-540.ckpt", map_location='cpu')
#     PuTransE.process_state_dict(state_dict)
#     PuTransE.load_state_dict(state_dict)
#
#     test_dataloader = TestDataLoader(train_dataloader.in_path, "link")
#     tester = Tester(model=PuTransE, data_loader=test_dataloader, use_gpu=torch.cuda.is_available())
#     # tester.run_link_prediction()
#     [data_head, data_tail] = test_dataloader.sampling_lp()
#
#     # Convert numpy keys to int
#     for universe in range(PuTransE.next_universe_id):
#         PuTransE.entity_id_mappings[universe] = defaultdict(int, {k.item(): v for k, v in
#                                                                   PuTransE.entity_id_mappings[universe].items()})
#         PuTransE.relation_id_mappings[universe] = defaultdict(int, {k.item(): v for k, v in
#                                                                     PuTransE.relation_id_mappings[universe].items()})
#
#     batch_h = data_head['batch_h']
#     batch_t = data_head['batch_t']
#     batch_r = data_head['batch_r']
#     mode = data_head['mode']
#
#     energy_scores = defaultdict(float)
#     entity_occurences = PuTransE.entity_universes[batch_t[0]] if mode == "head_batch" else PuTransE.entity_universes[
#         batch_h[0]]
#     relation_occurences = PuTransE.relation_universes[batch_r[0]]
#
#     # Gather embedding spaces in which the tuple is hold
#     embedding_space_ids = entity_occurences.intersection(relation_occurences)
#
#     # for loop but for test purposes selected first in embedding space ids
#     embedding_space_id = list(embedding_space_ids)[0]
#     # Get list with entities which are embedded in this space
#     embedding_space_entities = list(PuTransE.entity_id_mappings[embedding_space_id].keys())
#     # Calculate scores with embedding_space.predict({batch_h,batch_r,batch_t, mode})
#     embedding_space = PuTransE.trained_embedding_spaces[embedding_space_id]
#
#     batch_h = embedding_space_entities if mode == "head_batch" else list(batch_h)
#     batch_h = [PuTransE.entity_id_mappings[embedding_space_id][global_entity_id] for global_entity_id in batch_h]
#     batch_t = list(batch_t) if mode == "head_batch" else embedding_space_entities
#     batch_t = [PuTransE.entity_id_mappings[embedding_space_id][global_entity_id] for global_entity_id in batch_t]
#     batch_r = list(batch_r)
#     batch_r = [PuTransE.relation_id_mappings[embedding_space_id][global_relation_id] for global_relation_id in batch_r]
#
#     embedding_space_scores = embedding_space(
#         {"batch_h": PuTransE.to_tensor(batch_h, use_gpu=PuTransE.use_gpu),
#          "batch_t": PuTransE.to_tensor(batch_t, use_gpu=PuTransE.use_gpu),
#          "batch_r": PuTransE.to_tensor(batch_r, use_gpu=PuTransE.use_gpu),
#          "mode": "tail_batch"
#          }
#     )
#
#     for index, entity in enumerate(embedding_space_entities):
#         entity_score = embedding_space_scores[index]
#         if entity_score > energy_scores[entity]:
#             energy_scores[entity] = entity_score
#
#     energy_scores = 0

if False:
    # procedure to merge seperately learned PuTransE models
    PuTransE = ParallelUniverse(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
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
        norm=1,
        embedding_method="TransE",
        checkpoint_dir="./checkpoint/",
        save_steps=2)

    for model in PuTransE_learned_models:
        PuTransE.next_universe_id += model.next_universe_id
        for model_next_universe_id in range(model.next_universe_id):

            PuTransE.trained_embedding_spaces[PuTransE.next_universe_id + model_next_universe_id] = \
                model.trained_embedding_spaces[model_next_universe_id]

            for entity_key in list(model.entity_id_mappings[model_next_universe_id].keys()):
                PuTransE.entity_id_mappings[PuTransE.next_universe_id + model_next_universe_id][entity_key] = \
                    model.entity_id_mappings[model_next_universe_id][entity_key]

            for relation_key in list(model.relation_id_mappings[model_next_universe_id].keys()):
                PuTransE.relation_id_mappings[PuTransE.next_universe_id + model_next_universe_id][relation_key] = \
                    model.relation_id_mappings[model_next_universe_id][relation_key]
                # universe_id -> global entity_id -> universe entity_id

            for entity in range(model.ent_tot):
                PuTransE.entity_universes[entity].update(model.entity_universes[entity])  # entity_id -> universe_id

            for relation in range(model.rel_tot):
                PuTransE.relation_universes[relation].update(
                    model.relation_universes[relation])  # entity_id -> universe_id


#if False:

###### IMPORT
# Global energy estimation
if False:
    from openke.module.model import ParallelUniverse
    from openke.data import TrainDataLoader, TestDataLoader
    from openke.config import Trainer, Tester
    from collections import defaultdict
    import torch

    ###### Setup TRAINDATALOADER
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/WN18/",
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=0,
        neg_ent=1,
        neg_rel=0)

    ###### Setup PUTRANSE by loading already trained state
    PuTransE = ParallelUniverse(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
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
        norm=1,
        embedding_method="TransE",
        checkpoint_dir="./checkpoint/",
        save_steps=2)

    state_dict = torch.load("./checkpoint/PuTransE_learned_spaces-540.ckpt", map_location='cpu')
    PuTransE.process_state_dict(state_dict)
    PuTransE.load_state_dict(state_dict)

    ###### Setup a single test batch based on test triple
    test_dataloader = TestDataLoader(train_dataloader.in_path, "link")
    tester = Tester(model=PuTransE, data_loader=test_dataloader, use_gpu=torch.cuda.is_available())
    # tester.run_link_prediction()
    [data_head, data_tail] = test_dataloader.sampling_lp()

    batch_h = data_head['batch_h']
    batch_t = data_head['batch_t']
    batch_r = data_head['batch_r']
    mode = data_head['mode']

    ###### Global energy estimation
    # Gather embedding spaces in which the tuple (entity, relation) is hold
    triple_entity = batch_t[0] if mode == "head_batch" else batch_h[0]
    triple_relation = batch_r[0]
    evaluation_entities = batch_h if mode == "head_batch" else batch_t

    # Gather embedding spaces in which the tuple (entity, relation) is hold
    entity_occurences = PuTransE.entity_universes[triple_entity]
    relation_occurences = PuTransE.relation_universes[triple_relation]
    #embedding_space_ids = entity_occurences.intersection(relation_occurences)
    h = 271
    r = 0
    t = 17561
    head_occurences = PuTransE.entity_universes[271]
    tail_occurences = PuTransE.entity_universes[17561]
    rel_occurences = PuTransE.relation_universes[0]
    embedding_space_ids = head_occurences.intersection(rel_occurences.intersection(tail_occurences))

    energy_scores_dict = defaultdict(float)
    for entity in evaluation_entities:
        energy_scores_dict[entity] = -1.0  # if PuTransE.missing_embedding_handling == "last_rank" else

    for embedding_space_id in embedding_space_ids:
        # Get list with entities which are embedded in this space
        embedding_space_entities = list(PuTransE.entity_id_mappings[embedding_space_id].keys())
        embedding_space = PuTransE.trained_embedding_spaces[embedding_space_id]

        local_batch_h = embedding_space_entities if mode == "head_batch" else list(batch_h)
        local_batch_h = [PuTransE.entity_id_mappings[embedding_space_id][global_entity_id] for global_entity_id in
                         local_batch_h]
        local_batch_t = list(batch_t) if mode == "head_batch" else embedding_space_entities
        local_batch_t = [PuTransE.entity_id_mappings[embedding_space_id][global_entity_id] for global_entity_id in
                         local_batch_t]
        local_batch_r = batch_r[0]
        local_batch_r = [PuTransE.relation_id_mappings[embedding_space_id][local_batch_r]]

        embedding_space_scores = embedding_space.predict(
            {"batch_h": PuTransE.to_tensor(local_batch_h, use_gpu=PuTransE.use_gpu),
             "batch_t": PuTransE.to_tensor(local_batch_t, use_gpu=PuTransE.use_gpu),
             "batch_r": PuTransE.to_tensor(local_batch_r, use_gpu=PuTransE.use_gpu),
             "mode": mode
             }
        )
        embedding_space_scores = embedding_space_scores
        # iterate through dict and score tensor (index of both is equal) and transmit scores with comparison to energy_scores
        for index, entity in enumerate(embedding_space_entities):
            entity_score = embedding_space_scores[index]
            if entity_score > energy_scores_dict[entity]:
                energy_scores_dict[entity] = entity_score

    # Look how many missing values
    # list(energy_scores_dict.values()).count(-1) / len(energy_scores_dict)

    #np.array([energy_scores_dict[i] for i in batch_h if mode == "head_batch" else batch_t])
    energy_scores_dict = energy_scores_dict

# test rank
if False:
    from openke.module.model import ParallelUniverse
    from openke.data import TrainDataLoader, TestDataLoader
    from openke.config import Trainer, Tester
    from collections import defaultdict
    import torch

    ###### Setup TRAINDATALOADER
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/WN18/",
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=0,
        neg_ent=1,
        neg_rel=0)

    ###### Setup PUTRANSE by loading already trained state
    PuTransE = ParallelUniverse(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
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
        norm=1,
        embedding_method="TransE",
        checkpoint_dir="./checkpoint/",
        save_steps=2)

    state_dict = torch.load("./checkpoint/PuTransE_learned_spaces-540.ckpt", map_location='cpu')
    PuTransE.process_state_dict(state_dict)
    PuTransE.load_state_dict(state_dict)

    ###### Setup a single test batch based on test triple
    test_dataloader = TestDataLoader(train_dataloader.in_path, "link")
    tester = Tester(model=PuTransE, data_loader=test_dataloader, use_gpu=torch.cuda.is_available())

    # tester.run_link_prediction()
    [data_head, data_tail] = test_dataloader.sampling_lp()

    h = data_tail['batch_h'][0]
    t = data_head['batch_t'][0]
    r = data_head['batch_r'][0]
    mode = data_head['mode']
    print(h, r, t)

    score = PuTransE.predict({
        'batch_h': to_var(data_tail['batch_h'], False),
        'batch_t': to_var(data_tail['batch_t'], False),
        'batch_r': to_var(data_tail['batch_r'], False),
        'mode': data_tail['mode']
        })
    #Python ranking
    tail_scores = PuTransE.predict(data_tail)
    tail_scores[tail_scores == -1.0] = tail_scores.max()
    rank_tail = len(tail_scores[tail_scores < tail_scores[t]])

    head_scores = PuTransE.predict(data_head)
    head_scores[head_scores == -1.0] = head_scores.max()
    rank_head = len(head_scores[head_scores < head_scores[h]])

    tail_scores_c = PuTransE.predict(data_tail)
    tester.lib.testTail(tail_scores_c.__array_interface__["data"][0], 0, False)
    head_scores_c = PuTransE.predict(data_head)
    tester.lib.testHead(head_scores_c.__array_interface__["data"][0], 0, False)

    tester.lib.test_link_prediction(False)
    mr = tester.lib.getTestLinkMR(False)*test_dataloader.get_triple_tot()

# Examined Low rank of triple (21609, 36102, 0) after 1 universe 1 epoch
if False:
    from openke.module.model import ParallelUniverse
    from openke.data import TrainDataLoader, TestDataLoader
    from openke.config import Trainer, Tester
    from collections import defaultdict
    import torch
    from random import seed, randint

    # init_random_seed = randint(0, 2147483647)
    init_random_seed = 4

    print("Initial random seed is:", init_random_seed)

    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/WN18/",
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=0,
        neg_ent=1,
        neg_rel=0,
        initial_random_seed=init_random_seed)

    PuTransE = ParallelUniverse(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        train_dataloader=train_dataloader,
        initial_num_universes=5000,
        min_margin=1,
        max_margin=4,
        min_lr=0.01,
        max_lr=0.1,
        const_num_epochs=1,
        min_triple_constraint=500,
        max_triple_constraint=2000,
        balance=0.5,
        num_dim=50,
        norm=1,
        embedding_method="TransE",
        checkpoint_dir="./checkpoint/",
        save_steps=2)

    state_dict = torch.load("./checkpoint/1univ_1epoch.ckpt", map_location='cpu')
    PuTransE.process_state_dict(state_dict)
    PuTransE.load_state_dict(state_dict)

    test_dataloader = TestDataLoader(train_dataloader.in_path, "link")
    tester = Tester(model=PuTransE, data_loader=test_dataloader, use_gpu=torch.cuda.is_available())

    for index, [dh, dt] in enumerate(test_dataloader):
        idx = index
        data_head = dh
        data_tail = dt
        #21609, 36102, 0
        if (dt['batch_h'][0] == 21609 and dh['batch_t'][0] == 36102 and dt['batch_r'][0] == 0):
            break

    head_score = PuTransE.global_energy_estimation(data_head)
    head_score[head_score == -1.0] = head_score.max()
    rank_head = len(head_score[head_score < head_score[data_tail['batch_h'][0]]])

    head_score_c = PuTransE.global_energy_estimation(data_head)
    tester.lib.testHead(head_score_c.__array_interface__["data"][0], idx, False)

