import torch
import os
from random import randrange, uniform
from .Model import Model
from .TransE import TransE
from ...data import TrainDataLoader, TestDataLoader
from ...config import Trainer, Tester
from ..strategy import NegativeSampling
from ..loss import MarginLoss
from collections import defaultdict
import numpy as np


def defaultdict_int():
    return defaultdict(int)


class ParallelUniverse(Model):
    def __init__(self, ent_tot, rel_tot, use_gpu=False, train_dataloader=None, initial_num_universes=5000, min_margin=1,
                 max_margin=4, min_lr=0.01, max_lr=0.1, min_num_epochs=50, max_num_epochs=200,
                 min_triple_constraint=500,
                 max_triple_constraint=2000, balance=0.5, num_dim=50, norm=None, embedding_method="TransE",
                 save_steps=5, checkpoint_dir="./checkpoint/"):
        super(ParallelUniverse, self).__init__(ent_tot, rel_tot)
        self.use_gpu = use_gpu

        self.initial_num_universes = initial_num_universes
        self.next_universe_id = 0

        self.min_margin = min_margin
        self.max_margin = max_margin
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_num_epochs = min_num_epochs
        self.max_num_epochs = max_num_epochs
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

    def model_factory(self, embedding_method, ent_tot, rel_tot, dim, p_norm=1, norm_flag=True, margin=None,
                      epsilon=None):
        if embedding_method == "TransE":
            embedding_method = TransE(ent_tot, rel_tot, dim, p_norm, norm_flag, epsilon)

        if embedding_method == "":
            return

        return NegativeSampling(
            model=embedding_method,
            loss=MarginLoss(margin),
            batch_size=self.train_dataloader.get_batch_size()
        )

    def process_universe_mappings(self):
        entity_remapping, relation_remapping = self.train_dataloader.get_universe_mappings()

        entity_total_universe = self.train_dataloader.lib.getEntityTotalUniverse()
        for entity in range(entity_total_universe):
            self.entity_universes[entity_remapping[entity]].add(self.next_universe_id)
            self.entity_id_mappings[self.next_universe_id][entity_remapping[entity]] = entity

        relation_total_universe = self.train_dataloader.lib.getRelationTotalUniverse()
        for relation in range(relation_total_universe):
            self.relation_universes[relation_remapping[relation]].add(self.next_universe_id)
            self.relation_id_mappings[self.next_universe_id][relation_remapping[relation]] = relation

    def compile_train_datset(self):
        triple_constraint = randrange(self.min_triple_constraint, self.max_triple_constraint)
        balance_param = self.balance
        relation_in_focus = randrange(0, self.train_dataloader.relTotal - 1)

        print("universe information-------------------")
        print("--- num of training triples: %d" % triple_constraint)
        print("--- semantic focus is relation: %d" % relation_in_focus)
        self.train_dataloader.lib.getParallelUniverse(triple_constraint, balance_param, relation_in_focus)
        self.process_universe_mappings()
        print("--- num of universe entities: %d" % self.train_dataloader.lib.getEntityTotalUniverse())
        print("--- num of universe relations: %d" % self.train_dataloader.lib.getRelationTotalUniverse())
        print("---------------------------------------")

        print("Train dataset for embedding space compiled.")

    def train_embedding_space(self):
        # Create train dataset for universe and process mapping of contained global entities and relations
        self.compile_train_datset()

        # Create Model with factory
        entity_total_universe = self.train_dataloader.lib.getEntityTotalUniverse()
        relation_total_universe = self.train_dataloader.lib.getRelationTotalUniverse()
        margin = randrange(self.min_margin, self.max_margin)
        model = self.model_factory(embedding_method=self.embedding_method, ent_tot=entity_total_universe,
                                   rel_tot=relation_total_universe,
                                   dim=self.num_dim, p_norm=self.norm, margin=margin)
        # Initialize Trainer
        train_times = randrange(self.min_num_epochs, self.max_num_epochs)
        lr = round(uniform(self.min_lr, self.max_lr), len(str(self.min_lr).split('.')[1]))
        trainer = Trainer(model=model, data_loader=self.train_dataloader, train_times=train_times, alpha=lr,
                          use_gpu=self.use_gpu, opt_method="Adagrad")

        print("hyperparams for universe %d------------" % self.next_universe_id)
        print("--- epochs: %d" % train_times)
        print("--- learning rate:", lr)
        print("--- margin: %d" % margin)
        print("--- norm: %d" % self.norm)
        print("--- dimensions: %d" % self.num_dim)

        # Train embedding space
        self.train_dataloader.lib.swapHelpers()
        trainer.run()
        self.train_dataloader.lib.resetUniverse()

        return model.model

    def add_embedding_space(self, embedding_space):
        for param in embedding_space.parameters():
            param.requires_grad = False

        self.trained_embedding_spaces[self.next_universe_id] = embedding_space
        self.next_universe_id += 1

    def train_parallel_universes(self, num_of_embedding_spaces):
        for universe_id in range(num_of_embedding_spaces):
            embedding_space = self.train_embedding_space()
            self.add_embedding_space(embedding_space)

            if self.save_steps and self.checkpoint_dir and (universe_id + 1) % self.save_steps == 0:
                print("Learned %d universes." % self.next_universe_id)
                self.save_parameters(
                    os.path.join(self.checkpoint_dir + "Pu" + str(self.embedding_method) + "_learned_spaces-" +
                                 str(self.next_universe_id) + ".ckpt"))

    def to_tensor(self, x, use_gpu):
        if use_gpu:
            return torch.tensor([x]).cuda()
        else:
            return torch.tensor([x])

    def predict_triple(self, head_id, rel_id, tail_id):
        head_occurences = self.entity_universes[head_id]
        tail_occurences = self.entity_universes[tail_id]
        rel_occurences = self.relation_universes[rel_id]

        # Gather embedding spaces in which the triple is hold
        embedding_space_ids = head_occurences.intersection(tail_occurences.intersection(rel_occurences))

        # Iterate through spaces and get collect the max energy score
        max_energy_score = -1
        for embedding_space_id in embedding_space_ids:
            embedding_space = self.trained_embedding_spaces[embedding_space_id]

            local_head_id = self.entity_id_mappings[embedding_space_id][head_id]
            local_head_id = self.to_tensor(local_head_id, self.use_gpu)

            local_tail_id = self.entity_id_mappings[embedding_space_id][head_id]
            local_tail_id = self.to_tensor(local_tail_id, self.use_gpu)

            local_rel_id = self.relation_id_mappings[embedding_space_id][rel_id]
            local_rel_id = self.to_tensor(local_rel_id, self.use_gpu)

            energy_score = embedding_space(
                {"batch_h": local_head_id,
                 "batch_t": local_tail_id,
                 "batch_r": local_rel_id,
                 "mode": "normal"
                 }
            )

            if energy_score > max_energy_score:
                max_energy_score = energy_score

        return max_energy_score

    def forward(self):

        return 0

    def predict(self):
        return 0

    def extend_state_dict(self):
        state_dict = self.state_dict()
        state_dict.update(
            {"initial_num_universes": self.initial_num_universes,
             "next_universe_id": self.next_universe_id,
             "trained_embedding_spaces": self.trained_embedding_spaces,
             "entity_id_mappings": self.entity_id_mappings,
             "relation_id_mappings": self.relation_id_mappings,
             "entity_universes": self.entity_universes,
             "relation_universes": self.relation_universes,
             "min_margin": self.min_margin,
             "max_margin": self.max_margin,
             "min_lr": self.min_lr,
             "max_lr": self.max_lr,
             "min_num_epochs": self.min_num_epochs,
             "max_num_epochs": self.max_num_epochs,
             "min_triple_constraint": self.min_triple_constraint,
             "max_triple_constraint": self.max_triple_constraint,
             "balance": self.balance,
             "num_dim": self.num_dim,
             "norm": self.norm,
             "embedding_method": self.embedding_method})
        return state_dict

    def process_state_dict(self, state_dict):
        self.initial_num_universes = state_dict.pop("initial_num_universes")
        self.next_universe_id = state_dict.pop("next_universe_id")
        self.trained_embedding_spaces = state_dict.pop("trained_embedding_spaces")
        self.entity_id_mappings = state_dict.pop("entity_id_mappings")
        self.relation_id_mappings = state_dict.pop("relation_id_mappings")
        self.entity_universes = state_dict.pop("entity_universes")
        self.relation_universes = state_dict.pop("relation_universes")
        self.min_margin = state_dict.pop("min_margin")
        self.max_margin = state_dict.pop("max_margin")
        self.min_lr = state_dict.pop("min_lr")
        self.max_lr = state_dict.pop("max_lr")
        self.min_num_epochs = state_dict.pop("min_num_epochs")
        self.max_num_epochs = state_dict.pop("max_num_epochs")
        self.min_triple_constraint = state_dict.pop("min_triple_constraint")
        self.max_triple_constraint = state_dict.pop("max_triple_constraint")
        self.balance = state_dict.pop("balance")
        self.num_dim = state_dict.pop("num_dim")
        self.norm = state_dict.pop("norm")
        self.embedding_method = state_dict.pop("embedding_method")

    def save_parameters(self, path):
        state_dict = self.extend_state_dict()
        torch.save(state_dict, path)

    def load_parameters(self, path):
        state_dict = torch.load(self.checkpoint_dir + path)
        self.process_state_dict(state_dict)
        self.load_state_dict(state_dict)
