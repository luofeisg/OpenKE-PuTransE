import torch
import os
import ctypes
from random import randrange, uniform, seed
import numpy as np
from .Model import Model
from .TransE import TransE
from ...config import Trainer
from ...data import TestDataLoader
from ..strategy import NegativeSampling
from ..loss import MarginLoss
from collections import defaultdict
from tqdm import tqdm
from typing import Dict


def defaultdict_int():
    return defaultdict(int)


class ParallelUniverse(Model):
    def __init__(self, ent_tot, rel_tot, train_dataloader=None, initial_num_universes=5000, min_margin=1,
                 max_margin=4, min_lr=0.01, max_lr=0.1, min_num_epochs=50, max_num_epochs=200, const_num_epochs=None,
                 min_triple_constraint=500,
                 max_triple_constraint=2000, balance=0.5, num_dim=50, norm=None, embedding_method='TransE',
                 missing_embedding_handling='last_rank',
                 save_steps=5, checkpoint_dir='./checkpoint/', valid_steps=5):
        super(ParallelUniverse, self).__init__(ent_tot, rel_tot)
        self.use_gpu = torch.cuda.is_available()

        self.initial_num_universes = initial_num_universes
        self.next_universe_id = 0

        self.min_margin = min_margin
        self.max_margin = max_margin
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_num_epochs = min_num_epochs
        self.max_num_epochs = max_num_epochs
        self.const_num_epochs = const_num_epochs
        self.min_triple_constraint = min_triple_constraint
        self.max_triple_constraint = max_triple_constraint
        self.balance = balance

        self.num_dim = num_dim
        self.norm = norm
        self.embedding_method = embedding_method
        self.train_dataloader = train_dataloader

        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir

        self.trained_embedding_spaces = defaultdict(Model)
        self.entity_id_mappings = defaultdict(
            defaultdict_int)  # universe_id -> global entity_id -> universe entity_id
        self.relation_id_mappings = defaultdict(
            defaultdict_int)  # universe_id -> global relation_id -> universe relation_id

        self.entity_universes = defaultdict(set)  # entity_id -> universe_id
        self.relation_universes = defaultdict(set)  # relation_id -> universe_id

        self.initial_random_seed = self.train_dataloader.lib.getRandomSeed()

        """ Eval """
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)

        self.lib.validHead.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        self.lib.validTail.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        self.lib.getValidHit10.restype = ctypes.c_float
        self.valid_dataloader = TestDataLoader(train_dataloader.in_path, "link", mode='valid')

        self.missing_embedding_handling = missing_embedding_handling
        self.valid_steps = valid_steps
        self.early_stopping_patience = 10
        self.bad_counts = 0
        self.best_hit10 = 0

    def set_random_seed(self, rand_seed):
        self.train_dataloader.lib.setRandomSeed(rand_seed)
        self.train_dataloader.lib.randReset()
        seed(rand_seed)

    def set_valid_dataloader(self, valid_dataloader):
        self.valid_dataloader = valid_dataloader

    def model_factory(self, embedding_method, ent_tot, rel_tot, dim, p_norm=1, norm_flag=True, margin=None,
                      epsilon=None, batch_size=None):
        if embedding_method == 'TransE':
            embedding_method = TransE(ent_tot, rel_tot, dim, p_norm, norm_flag, epsilon)

        if embedding_method == '':
            return

        return NegativeSampling(
            model=embedding_method,
            loss=MarginLoss(margin),
            batch_size=self.train_dataloader.batch_size
        )

    def valid(self):
        self.lib.validInit()
        validation_range = tqdm(self.valid_dataloader)
        for index, [valid_head_batch, valid_tail_batch] in enumerate(validation_range):
            score = self.predict(valid_head_batch)
            self.lib.validHead(score.__array_interface__["data"][0], index)
            score = self.predict(valid_tail_batch)
            self.lib.validTail(score.__array_interface__["data"][0], index)
        return self.lib.getValidHit10()

    def process_universe_mappings(self):
        entity_remapping, relation_remapping = self.train_dataloader.get_universe_mappings()

        entity_total_universe = self.train_dataloader.lib.getEntityTotalUniverse()
        print('Entities are %d' % (entity_total_universe))
        for entity in range(entity_total_universe):
            self.entity_universes[entity_remapping[entity].item()].add(self.next_universe_id)
            self.entity_id_mappings[self.next_universe_id][entity_remapping[entity].item()] = entity

        relation_total_universe = self.train_dataloader.lib.getRelationTotalUniverse()
        for relation in range(relation_total_universe):
            self.relation_universes[relation_remapping[relation].item()].add(self.next_universe_id)
            self.relation_id_mappings[self.next_universe_id][relation_remapping[relation].item()] = relation

    def compile_train_datset(self):
        # Create train dataset for universe and process mapping of contained global entities and relations
        triple_constraint = randrange(self.min_triple_constraint, self.max_triple_constraint)
        balance_param = self.balance
        relation_in_focus = randrange(0, self.train_dataloader.relTotal - 1)

        print('universe information-------------------')
        print('--- num of training triples: %d' % triple_constraint)
        print('--- semantic focus is relation: %d' % relation_in_focus)
        self.train_dataloader.compile_universe_dataset(triple_constraint, balance_param, relation_in_focus)
        self.process_universe_mappings()
        print('--- num of universe entities: %d' % self.train_dataloader.lib.getEntityTotalUniverse())
        print('--- num of universe relations: %d' % self.train_dataloader.lib.getRelationTotalUniverse())
        print('---------------------------------------')

        print('Train dataset for embedding space compiled.')

    def train_embedding_space(self):
        # Create Model with factory
        entity_total_universe = self.train_dataloader.lib.getEntityTotalUniverse()
        relation_total_universe = self.train_dataloader.lib.getRelationTotalUniverse()
        margin = randrange(self.min_margin, self.max_margin)
        model = self.model_factory(embedding_method=self.embedding_method, ent_tot=entity_total_universe,
                                   rel_tot=relation_total_universe,
                                   dim=self.num_dim, p_norm=self.norm, margin=margin)
        # Initialize Trainer
        train_times = self.const_num_epochs if self.const_num_epochs else randrange(self.min_num_epochs, self.max_num_epochs)
        lr = round(uniform(self.min_lr, self.max_lr), len(str(self.min_lr).split('.')[1]))
        trainer = Trainer(model=model, data_loader=self.train_dataloader, train_times=train_times, alpha=lr,
                          use_gpu=self.use_gpu, opt_method='Adagrad')

        print('hyperparams for universe %d------------' % self.next_universe_id)
        print('--- epochs: %d' % train_times)
        print('--- learning rate:', lr)
        print('--- margin: %d' % margin)
        print('--- norm: %d' % self.norm)
        print('--- dimensions: %d' % self.num_dim)

        # Train embedding space
        self.train_dataloader.swap_helpers()
        trainer.run()
        self.train_dataloader.reset_universe()

        return model.model

    def add_embedding_space(self, embedding_space):
        for param in embedding_space.parameters():
            param.requires_grad = False

        self.trained_embedding_spaces[self.next_universe_id] = embedding_space

    def train_parallel_universes(self, num_of_embedding_spaces):
        for universe_id in range(num_of_embedding_spaces):
            self.set_random_seed(self.initial_random_seed + self.next_universe_id)
            self.compile_train_datset()
            embedding_space = self.train_embedding_space()
            self.add_embedding_space(embedding_space)
            self.next_universe_id += 1

            if (universe_id + 1) % self.valid_steps == 0:
                print("Universe %d has finished, validating..." % (universe_id))
                hit10 = self.valid()
                if hit10 > self.best_hit10:
                    best_hit10 = hit10
                    print("Best model | hit@10 of valid set is %f" % (best_hit10))
                    self.bad_counts = 0
                else:
                    print(
                        "Hit@10 of valid set is %f | bad count is %d"
                        % (hit10, self.bad_counts)
                    )
                    self.bad_counts += 1
                if self.bad_counts == self.early_stopping_patience:
                    print("Early stopping at universe %d" % (universe_id))
                    break

            if self.save_steps and self.checkpoint_dir and (universe_id + 1) % self.save_steps == 0:
                print('Learned %d universes.' % self.next_universe_id)
                self.save_parameters(
                    os.path.join(self.checkpoint_dir + 'Pu' + str(self.embedding_method) + '_learned_spaces-' +
                                 str(self.next_universe_id) + '.ckpt'))

    def to_tensor(self, x, use_gpu):
        if use_gpu:
            return torch.tensor([x]).cuda()
        else:
            return torch.tensor([x])

    def predict_triple(self, head_id, rel_id, tail_id, mode='normal'):
        head_occurences = self.entity_universes[head_id]
        tail_occurences = self.entity_universes[tail_id]
        rel_occurences = self.relation_universes[rel_id]

        # Gather embedding spaces in which the triple is hold
        embedding_space_ids = head_occurences.intersection(tail_occurences.intersection(rel_occurences))

        # Iterate through spaces and get collect the max energy score
        max_energy_score = -1.0
        for embedding_space_id in embedding_space_ids:
            embedding_space = self.trained_embedding_spaces[embedding_space_id]

            local_head_id = self.entity_id_mappings[embedding_space_id][head_id]
            local_head_id = self.to_tensor(local_head_id, self.use_gpu)
            local_tail_id = self.entity_id_mappings[embedding_space_id][tail_id]
            local_tail_id = self.to_tensor(local_tail_id, self.use_gpu)
            local_rel_id = self.relation_id_mappings[embedding_space_id][rel_id]
            local_rel_id = self.to_tensor(local_rel_id, self.use_gpu)

            energy_score = embedding_space.predict(
                {'batch_h': local_head_id,
                 'batch_t': local_tail_id,
                 'batch_r': local_rel_id,
                 'mode': mode
                 }
            )

            if energy_score > max_energy_score:
                max_energy_score = energy_score

        return max_energy_score

    def global_energy_estimation(self, data):
        batch_h = data['batch_h'].numpy() if type(data['batch_h']) == torch.Tensor else data['batch_h']
        batch_t = data['batch_t'].numpy() if type(data['batch_t']) == torch.Tensor else data['batch_t']
        batch_r = data['batch_r'].numpy() if type(data['batch_r']) == torch.Tensor else data['batch_r']
        mode = data['mode']

        triple_entity = batch_t[0] if mode == 'head_batch' else batch_h[0]
        triple_relation = batch_r[0]
        evaluation_entities = batch_h if mode == 'head_batch' else batch_t

        # Gather embedding spaces in which the tuple (entity, relation) is hold
        entity_occurences = self.entity_universes[triple_entity]
        relation_occurences = self.relation_universes[triple_relation]
        embedding_space_ids = entity_occurences.intersection(relation_occurences)

        energy_scores_dict = defaultdict(float)
        for entity in evaluation_entities:
            energy_scores_dict[entity] = -1.0  # if self.missing_embedding_handling == 'last_rank' else

        for embedding_space_id in embedding_space_ids:
            # Get list with entities which are embedded in this space
            embedding_space_entity_dict = self.entity_id_mappings[embedding_space_id]
            # Calculate scores with embedding_space.predict({batch_h,batch_r,batch_t, mode})
            embedding_space = self.trained_embedding_spaces[embedding_space_id]

            local_batch_h = embedding_space_entity_dict.keys() if mode == "head_batch" else list(batch_h)
            local_batch_h = [self.entity_id_mappings[embedding_space_id][global_entity_id] for global_entity_id in
                             local_batch_h]
            local_batch_t = list(batch_t) if mode == "head_batch" else embedding_space_entity_dict.keys()
            local_batch_t = [self.entity_id_mappings[embedding_space_id][global_entity_id] for global_entity_id in
                             local_batch_t]
            local_batch_r = list(batch_r)
            local_batch_r = [self.relation_id_mappings[embedding_space_id][global_relation_id] for global_relation_id in
                             local_batch_r]

            embedding_space_scores = embedding_space.predict(
                {"batch_h": self.to_tensor(local_batch_h, use_gpu=self.use_gpu),
                 "batch_t": self.to_tensor(local_batch_t, use_gpu=self.use_gpu),
                 "batch_r": self.to_tensor(local_batch_r, use_gpu=self.use_gpu),
                 "mode": mode
                 }
            )
            # iterate through dict and score tensor (index of both is equal) and transmit scores with comparison to energy_scores
            for global_entity_id, local_entity_id in embedding_space_entity_dict.items():
                entity_score = embedding_space_scores[local_entity_id]
                if entity_score > energy_scores_dict[global_entity_id]:
                    energy_scores_dict[global_entity_id] = entity_score

            # return np.array([energy_scores_dict[i] for i in batch_h if mode == "head_batch" else batch_t])
        return np.fromiter(energy_scores_dict.values(), dtype=np.float32)

    def forward(self, data: dict):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']

        if mode == 'head_batch' or mode == 'tail_batch':
            score = self.global_energy_estimation(data)
        elif mode == 'normal':
            num_of_scores = batch_h.size()[0]
            score = torch.zeros(num_of_scores)
            for index in range(num_of_scores):
                score[index] = self.predict_triple(batch_h[index], batch_r[index], batch_t[index], mode)

        return score

    def predict(self, data: dict):
        score = self.forward(data)
        return score

    def extend_parallel_universe(self, ParallelUniverse_inst):
        # shift indexes of trained embedding spaces in parameter instance to add them to this instance
        for universe_id in list(ParallelUniverse_inst.trained_embedding_spaces.keys()):
            ParallelUniverse_inst.trained_embedding_spaces[
                universe_id + self.next_universe_id] = ParallelUniverse_inst.trained_embedding_spaces.pop(universe_id)
        self.trained_embedding_spaces.update(ParallelUniverse_inst.trained_embedding_spaces)

        for entity in range(ParallelUniverse_inst.ent_tot):
            self.entity_universes[entity].update(
                ParallelUniverse_inst.entity_universes[entity])  # entity_id -> universe_id

        for relation in range(ParallelUniverse_inst.rel_tot):
            self.relation_universes[relation].update(
                ParallelUniverse_inst.relation_universes[relation])  # entity_id -> universe_id

        for instance_next_universe_id in range(ParallelUniverse_inst.next_universe_id):
            for entity_key in list(ParallelUniverse_inst.entity_id_mappings[instance_next_universe_id].keys()):
                self.entity_id_mappings[self.next_universe_id + instance_next_universe_id][entity_key] = \
                    ParallelUniverse_inst.entity_id_mappings[instance_next_universe_id][entity_key]

            for relation_key in list(ParallelUniverse_inst.relation_id_mappings[instance_next_universe_id].keys()):
                self.relation_id_mappings[self.next_universe_id + instance_next_universe_id][relation_key] = \
                    ParallelUniverse_inst.relation_id_mappings[instance_next_universe_id][relation_key]
                # universe_id -> global entity_id -> universe entity_id

        self.next_universe_id += ParallelUniverse_inst.next_universe_id

    def extend_state_dict(self):
        state_dict = self.state_dict()
        state_dict.update(
            {'initial_num_universes': self.initial_num_universes,
             'next_universe_id': self.next_universe_id,
             'trained_embedding_spaces': self.trained_embedding_spaces,
             'entity_id_mappings': self.entity_id_mappings,
             'relation_id_mappings': self.relation_id_mappings,
             'entity_universes': self.entity_universes,
             'relation_universes': self.relation_universes,
             'min_margin': self.min_margin,
             'max_margin': self.max_margin,
             'min_lr': self.min_lr,
             'max_lr': self.max_lr,
             'min_num_epochs': self.min_num_epochs,
             'max_num_epochs': self.max_num_epochs,
             'min_triple_constraint': self.min_triple_constraint,
             'max_triple_constraint': self.max_triple_constraint,
             'balance': self.balance,
             'num_dim': self.num_dim,
             'norm': self.norm,
             'embedding_method': self.embedding_method
             # 'best_hit10': self.best_hit10,
             # 'bad_counts': self.bad_counts
             })
        return state_dict

    def process_state_dict(self, state_dict):
        self.initial_num_universes = state_dict.pop('initial_num_universes')
        self.next_universe_id = state_dict.pop('next_universe_id')
        self.trained_embedding_spaces = state_dict.pop('trained_embedding_spaces')
        self.entity_id_mappings = state_dict.pop('entity_id_mappings')
        self.relation_id_mappings = state_dict.pop('relation_id_mappings')
        self.entity_universes = state_dict.pop('entity_universes')
        self.relation_universes = state_dict.pop('relation_universes')
        self.min_margin = state_dict.pop('min_margin')
        self.max_margin = state_dict.pop('max_margin')
        self.min_lr = state_dict.pop('min_lr')
        self.max_lr = state_dict.pop('max_lr')
        self.min_num_epochs = state_dict.pop('min_num_epochs')
        self.max_num_epochs = state_dict.pop('max_num_epochs')
        self.min_triple_constraint = state_dict.pop('min_triple_constraint')
        self.max_triple_constraint = state_dict.pop('max_triple_constraint')
        self.balance = state_dict.pop('balance')
        self.num_dim = state_dict.pop('num_dim')
        self.norm = state_dict.pop('norm')
        self.embedding_method = state_dict.pop('embedding_method')
        # self.best_hit10 = state_dict.pop('best_hit10')
        # self.bad_counts = state_dict.pop('bad_counts')

    def calculate_unembedded_ratio(self, mode='examine_entities'):
        num_unembedded = 0
        mapping_dict = self.entity_universes if mode == 'examine_entities' else self.relation_universes
        num_total = self.train_dataloader.entTotal if mode == 'examine_entities' else self.train_dataloader.relTotal

        for i in range(num_total):
            if len(mapping_dict[i]) == 0:
                num_unembedded += 1

        return num_unembedded / num_total

    def save_parameters(self, path):
        state_dict = self.extend_state_dict()
        torch.save(state_dict, path)

    def load_parameters(self, path):
        state_dict = torch.load(self.checkpoint_dir + path)
        self.process_state_dict(state_dict)
        self.load_state_dict(state_dict)
