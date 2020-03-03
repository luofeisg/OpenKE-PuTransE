import torch
import torch.nn as nn
from random import randrange, uniform
from .Model import Model
from .TransE import TransE
from ...data import TrainDataLoader, TestDataLoader
from ...config import Trainer, Tester
from ..strategy import NegativeSampling
from ..loss import MarginLoss
from collections import defaultdict
import numpy as np


class ParallelUniverse(Model):

    def __init__(self, ent_tot, rel_tot, use_gpu=False, train_dataloader=None, initial_num_universes=5000, min_margin=1,
                 max_margin=4,
                 min_lr=0.01, max_lr=0.1, min_num_epochs=50, max_num_epochs=200, min_triple_constraint=500,
                 max_triple_constraint=2000, balance=0.5, num_dim=50, norm=None, embedding_method="TransE"):
        super(ParallelUniverse, self).__init__(ent_tot, rel_tot)
        self.use_gpu = use_gpu

        self.initial_num_universes = initial_num_universes
        self.next_universe_id = 0

        self.trained_embedding_spaces = defaultdict(lambda: Model)
        self.entity_id_mappings = defaultdict(
            lambda: defaultdict(int))  # universe_id -> global entity_id -> universe entity_id
        self.relation_id_mappings = defaultdict(
            lambda: defaultdict(int))  # universe_id -> global relation_id -> universe relation_id

        self.entity_universes = defaultdict(set)  # entity_id -> universe_id
        self.relation_universes = defaultdict(set)  # relation_id -> universe_id

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

    def model_factory(self, embedding_method, ent_tot, rel_tot, dim, p_norm=1, norm_flag=True, margin=None, epsilon=None):
        if embedding_method == "TransE":
            embedding_method = TransE(ent_tot, rel_tot, dim, p_norm, norm_flag, margin, epsilon)

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
        print("--- num of universe entities: %d" % self.train_dataloader.lib.getEntityTotalUniverse())
        print("--- num of universe relations: %d" % self.train_dataloader.lib.getRelationTotalUniverse())
        print("--- semantic focus is relation: %d" % relation_in_focus)
        print("---------------------------------------")

        self.train_dataloader.lib.getParallelUniverse(triple_constraint, balance_param, relation_in_focus)
        self.process_universe_mappings()

        print("Train dataset for embedding space compiled.")

    def train_embedding_space(self):
        # Create train dataset for universe and process mapping of contained global entities and relations
        self.compile_train_datset()

        # Create Model with factory
        entity_total_universe = self.train_dataloader.lib.getEntityTotalUniverse()
        relation_total_universe = self.train_dataloader.lib.getRelationTotalUniverse()
        margin = randrange(self.min_margin, self.max_margin)
        model = self.model_factory(self.embedding_method, entity_total_universe, relation_total_universe,
                                   self.num_dim, self.norm, margin)
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
        print("universe information-------------------")
        print("--- num of training triples: %d" % triple_constraint)
        print("--- num of universe entities: %d" % entity_total_universe)
        print("--- num of universe relations: %d" % relation_total_universe)
        print("--- semantic focus is relation: %d" % relation_in_focus)
        print("---------------------------------------")

        # Train embedding space
        self.train_dataloader.lib.swapHelpers()
        trainer.run()
        self.train_dataloader.lib.resetUniverse()

        return model.model

    def add_embedding_space(self, embedding_space):
        self.trained_embedding_spaces[self.next_universe_id] = embedding_space
        self.next_universe_id += 1

    def train_parallel_universes(self):
        for universe_id in range(self.initial_num_universes):
            embedding_space = self.train_embedding_space()
            self.add_embedding_space(embedding_space)

    def forward(self):
        return 0

    def predict(self):
        return 0


