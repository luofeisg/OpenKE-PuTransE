import torch
import torch.nn as nn
from random import randrange
from .Model import Model
from .TransE import TransE
from ...data import TrainDataLoader, TestDataLoader
from ...config import Trainer, Tester
from ..strategy import NegativeSampling
from ..loss import MarginLoss
from typing import DefaultDict


class ParallelUniverse(Model):

    def __init__(self, use_gpu=False, in_path="./", initial_num_universes=5000, min_margin=1, max_margin=4,
                 min_lr=0.01, max_lr=0.1, min_num_epochs=50, max_num_epochs=200, min_triple_constraint=500,
                 max_triple_constraint=2000, balance=0.5, num_dim=50, norm=None, model_name="TransE"):
        self.use_gpu = use_gpu

        self.in_path = in_path;
        self.initial_num_universes = initial_num_universes

        self.min_triple_constraint = min_triple_constraint
        self.max_triple_constraint = max_triple_constraint
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
        self.model_name = model_name
        self.train_dataloader = TrainDataLoader(
            in_path="./benchmarks/FB15K237/",
            nbatches=100,
            threads=8,
            sampling_mode="normal",
            bern_flag=0,
            filter_flag=1,
            neg_ent=1,
            neg_rel=0)

    def model_factory(self, model_name, ent_tot, rel_tot, dim, p_norm=1, norm_flag=True, margin=None, epsilon=None):
        if model_name == "TransE":
            self.embedding_method = TransE(ent_tot, rel_tot, dim, p_norm, norm_flag, margin, epsilon)

        if model_name == "":
            return

        return NegativeSampling(
            model=self.embedding_method,
            loss=MarginLoss(margin),
            batch_size=self.train_dataloader.get_batch_size()
        )

    def createUniverse(self):
        triple_constraint = randrange(self.min_triple_constraint, self.max_triple_constraint)
        balance_param = self.balance
        relation_in_focus = randrange(0, self.train_dataloader.relTotal - 1)

        self.train_dataloader.lib.getParallelUniverse(triple_constraint, balance_param, relation_in_focus)

        # Process mapping of embedding space entities and relations

        # create Model with factory
        num_entities_in_universe = self.train_dataloader.lib.getEntityTotalUniverse()
        num_relations_in_universe = self.train_dataloader.lib.getRelationTotalUniverse()
        margin = randrange(self.min_margin, self.max_margin)
        model = self.model_factory(self.model_name, num_entities_in_universe, num_relations_in_universe,
                                   self.num_dim, self.norm, margin)
        # create Trainer
        train_times = randrange(self.min_num_epochs, self.max_num_epochs)
        lr = randrange(self.min_lr, self.max_lr)
        trainer = Trainer(model=model, data_loader=self.train_dataloader, train_times=train_times, alpha=lr,
                          use_gpu=self.use_gpu, opt_method = "Adagrad")

        # Train embedding space
        self.train_dataloader.lib.swapHelpers()
        trainer.run()
        self.train_dataloader.lib.resetUniverse()

        #


    def trainParallelUniverses(self):
        for universe in self.initial_num_universes:
            self.createUniverse()

    def predict(self, data):
        score = -self.forward(data)
        return 0
