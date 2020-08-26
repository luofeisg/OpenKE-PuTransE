import torch
import time
import os
import ctypes
from random import randrange, uniform, seed
import numpy as np
from ..module.model.Model import Model
from .Trainer import Trainer
from .Tester import Tester
from ..data import TestDataLoader
from ..module.strategy import NegativeSampling
from ..module.loss import MarginLoss
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path


def get_string_key(entity, relation):
    return '{},{}'.format(entity, relation)


def defaultdict_int(innerfactory=int):
    return defaultdict(innerfactory)


def float_default():
    return float("inf")


def to_tensor(x, use_gpu):
    if use_gpu:
        return torch.tensor([x]).cuda()
    else:
        return torch.tensor([x])


class Parallel_Universe_Config(Tester):
    def __init__(self,
                 train_dataloader=None, training_identifier='', valid_dataloader=None, test_dataloader=None,
                 initial_num_universes=5000,
                 min_margin=1, max_margin=4, min_lr=0.01, max_lr=0.1, min_num_epochs=50, max_num_epochs=200,
                 const_num_epochs=None, min_triple_constraint=500, max_triple_constraint=2000, min_balance=0.25,
                 max_balance=0.5, embedding_model=None, embedding_model_param=None,
                 missing_embedding_handling='last_rank',
                 save_steps=5, checkpoint_dir='./checkpoint/', valid_steps=5, early_stopping_patience=5,
                 training_setting="static",
                 incremental_strategy="normal"):
        super(Parallel_Universe_Config, self).__init__(data_loader=test_dataloader, use_gpu=torch.cuda.is_available())

        """ Train data + variables"""
        self.train_dataloader = train_dataloader
        self.ent_tot = train_dataloader.entTotal
        self.rel_tot = train_dataloader.relTotal
        self.training_identifier = training_identifier

        """ "-constant traininghyper parameters" """
        self.embedding_model = embedding_model
        self.embedding_model_param = embedding_model_param

        """ Parallel Universe data structures """
        self.initial_num_universes = initial_num_universes
        self.next_universe_id = 0

        self.trained_embedding_spaces = defaultdict(Model)  # universe_id -> embedding_space
        self.entity_id_mappings = defaultdict(defaultdict_int)  # universe_id -> global entity_id -> local_entity_id
        self.relation_id_mappings = defaultdict(
            defaultdict_int)  # universe_id -> global relation_id -> local_relation_id

        self.entity_universes = defaultdict(set)  # entity_id -> universe_id
        self.relation_universes = defaultdict(set)  # relation_id -> universe_id

        self.initial_random_seed = self.train_dataloader.lib.getRandomSeed()

        """Parallel Universe spans for randomizing embedding space hyper parameters"""
        self.min_margin = min_margin
        self.max_margin = max_margin
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_num_epochs = min_num_epochs
        self.max_num_epochs = max_num_epochs
        self.const_num_epochs = const_num_epochs
        self.min_triple_constraint = min_triple_constraint
        self.max_triple_constraint = max_triple_constraint
        self.min_balance = min_balance
        self.max_balance = max_balance

        """ saving """
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir

        """ Eval """
        self.missing_embedding_handling = missing_embedding_handling  # ["last_rank" | "null_vector" i.e. max of f(h,r,0) or f(0,r,t)]

        """ ""Valid"" """
        self.lib.validHead.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        self.lib.validTail.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        self.lib.getValidHit10.restype = ctypes.c_float

        self.valid_dataloader = valid_dataloader if valid_dataloader != None else TestDataLoader(
            train_dataloader.in_path,
            sampling_mode="link",
            mode='valid')

        self.valid_steps = valid_steps
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_patience_const = early_stopping_patience
        self.bad_counts = 0
        self.best_hit10 = 0
        self.best_state = None

        """Global Energy Estimation data structures"""
        self.current_tested_universes = 0
        self.current_validated_universes = 0
        self.evaluation_head2tail_triple_score_dict = {}
        self.evaluation_tail2head_triple_score_dict = {}
        self.evaluation_head2rel_tuple_score_dict = {}
        self.evaluation_tail2rel_tuple_score_dict = {}
        self.default_scores = [float("inf")] * self.ent_tot

        self.training_setting = training_setting  # ["incremental" | "static"]
        self.incremental_strategy = incremental_strategy  # ["normal" | "deprecate"]
        if self.training_setting == "incremental":
            self.deprecated_embeddingspaces = set()

    def get_default_value_list(self):
        return [float("inf") for i in range(self.ent_tot)]

    def set_min_max_triple_constraint(self, min, max):
        self.min_triple_constraint = min
        self.max_triple_constraint = max

    def set_random_seed(self, rand_seed):
        self.train_dataloader.lib.setRandomSeed(rand_seed)
        self.train_dataloader.lib.randReset()
        seed(rand_seed)
        torch.manual_seed(rand_seed)

    def set_valid_dataloader(self, valid_dataloader):
        self.valid_dataloader = valid_dataloader

    def set_test_dataloader(self, test_dataloader):
        self.data_loader = test_dataloader

    def embedding_model_factory(self, ent_tot, rel_tot, margin):
        embedding_method = self.embedding_model(ent_tot, rel_tot, **self.embedding_model_param)
        output_model = NegativeSampling(
            model=embedding_method,
            loss=MarginLoss(margin=margin),
            batch_size=self.train_dataloader.batch_size
        )

        return output_model

    def valid(self):
        self.lib.validInit()
        validation_range = tqdm(self.valid_dataloader)
        for index, [valid_head_batch, valid_tail_batch] in enumerate(validation_range):
            score = self.global_energy_estimation(valid_head_batch)
            self.lib.validHead(score.__array_interface__["data"][0], index)
            score = self.global_energy_estimation(valid_tail_batch)
            self.lib.validTail(score.__array_interface__["data"][0], index)
        return self.lib.getValidHit10()

    def reset_valid_variables(self):
        self.early_stopping_patience = self.early_stopping_patience_const
        self.best_state = {}
        self.best_hit10 = 0
        self.bad_counts = 0

    def process_universe_mappings(self):
        entity_remapping, relation_remapping = self.train_dataloader.get_universe_mappings()

        entity_total_universe = self.train_dataloader.lib.getEntityTotalUniverse()
        print('Entities are %d' % entity_total_universe)
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
        balance_param = round(uniform(self.min_balance, self.max_balance), 2)

        # Outsourced sampling of relation to C++ getParallelUniverse in file
        # UniverseConstructor.h (l.291 - l.299)
        # relation_in_focus = randrange(0, self.train_dataloader.relTotal - 1)

        print('universe information-------------------')
        print('--- num of training triples: %d' % triple_constraint)
        self.train_dataloader.compile_universe_dataset(triple_constraint, balance_param)
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
        model = self.embedding_model_factory(ent_tot=entity_total_universe, rel_tot=relation_total_universe,
                                             margin=margin)

        # Initialize Trainer
        train_times = self.const_num_epochs if self.const_num_epochs is not None \
            else randrange(self.min_num_epochs, self.max_num_epochs)

        lr = round(uniform(self.min_lr, self.max_lr), len(str(self.min_lr).split('.')[1]))
        trainer = Trainer(model=model, data_loader=self.train_dataloader, train_times=train_times, alpha=lr,
                          use_gpu=self.use_gpu, opt_method='Adagrad')

        print('hyperparams for universe %d------------' % self.next_universe_id)
        print('--- epochs: %d' % train_times)
        print('--- learning rate:', lr)
        print('--- margin: %d' % margin)
        if "p_norm" in self.embedding_model_param:
            print('--- norm: %d' % self.embedding_model_param['p_norm'])
        if "dim" in self.embedding_model_param:
            print('--- dimensions: %d' % self.embedding_model_param['dim'])

        # Train embedding space
        self.train_dataloader.swap_helpers()
        trainer.run()
        self.train_dataloader.reset_universe()

        return model.model

    def add_embedding_space(self, embedding_space):
        for param in embedding_space.parameters():
            param.requires_grad = False

        self.trained_embedding_spaces[self.next_universe_id] = embedding_space

    def save_model(self, filename=None):
        save_directory = self.checkpoint_dir
        if not filename:
            filename = "Pu{}_learned_spaces-{}_{}.ckpt".format(self.embedding_model.__name__,
                                                               self.next_universe_id,
                                                               self.training_identifier)
        file_directory = "{}{}".format(save_directory, filename)
        self.save_parameters(os.path.join(file_directory))

    def save_best_state(self):
        self.best_state = self.get_state()

    def get_state(self):
        # best_state = deepcopy(self)
        # best_state.evaluation_head2tail_triple_score_dict = None
        # best_state.evaluation_tail2head_triple_score_dict = None
        # best_state.evaluation_head2rel_tuple_score_dict = None
        # best_state.evaluation_tail2rel_tuple_score_dict = None

        state = {
            "trained_embedding_spaces": deepcopy(self.trained_embedding_spaces),
            "next_universe_id": self.next_universe_id,
            "entity_id_mappings": deepcopy(self.entity_id_mappings),
            "relation_id_mappings": deepcopy(self.relation_id_mappings),

            "entity_universes": deepcopy(self.entity_universes),
            "relation_universes": deepcopy(self.relation_universes),
        }
        return state

    def get_best_state(self):
        if self.best_state:
            print("Get best state...")
            print("switch {} with {} trained embedding spaces.".format(len(self.trained_embedding_spaces), len(
                self.best_state["trained_embedding_spaces"])))
            self.trained_embedding_spaces = self.best_state["trained_embedding_spaces"]

            self.entity_id_mappings = self.best_state["entity_id_mappings"]
            self.relation_id_mappings = self.best_state["relation_id_mappings"]
            self.entity_universes = self.best_state["entity_universes"]
            self.relation_universes = self.best_state["relation_universes"]

            print(
                "Trained {} universes but switch to best state with {} trained universes.".format(self.next_universe_id,
                                                                                                  self.best_state[
                                                                                                      "next_universe_id"]))
            self.next_universe_id = self.best_state["next_universe_id"]

        return self

    def train_parallel_universes(self, num_of_embedding_spaces):
        start_time = time.time()
        for universe_id in range(num_of_embedding_spaces):
            self.set_random_seed(self.initial_random_seed + self.next_universe_id)
            self.compile_train_datset()
            embedding_space = self.train_embedding_space()
            self.add_embedding_space(embedding_space)
            self.next_universe_id += 1

            if (universe_id + 1) % self.valid_steps == 0:
                print("Universe %d has finished, validating..." % (self.next_universe_id - 1))
                self.eval_universes(eval_mode='valid')
                hit10 = self.valid()
                print("Current hit@10: {}".format(hit10))
                if hit10 > self.best_hit10:
                    self.best_hit10 = hit10
                    print("Best model | hit@10 of valid set is %f" % self.best_hit10)
                    print('Save model at universe %d.' % self.next_universe_id)
                    self.save_model("Best_model_Pu{}_{}.ckpt".format(self.embedding_model.__name__,
                                                                     self.training_identifier))
                    self.bad_counts = 0
                    # self.save_best_state()
                else:
                    print(
                        "Hit@10 of valid set is %f | bad count is %d"
                        % (hit10, self.bad_counts)
                    )
                    self.bad_counts += 1
                if self.bad_counts == self.early_stopping_patience:
                    print("Early stopping at universe {}".format(self.next_universe_id - 1))
                    self.get_best_state()
                    break

            if self.save_steps and self.checkpoint_dir and (universe_id + 1) % self.save_steps == 0:
                print('Save model at universe %d.' % self.next_universe_id)
                self.save_model()
        end_time = time.time()
        print('Time took for creation of embedding spaces: {:5.3f}s'.format(end_time - start_time), end='  ')

    def gather_embedding_spaces(self, entity_1, rel, entity_2=None):
        entity_occurences = self.entity_universes[entity_1]
        relation_occurences = self.relation_universes[rel]
        embedding_space_ids = entity_occurences.intersection(relation_occurences)
        if entity_2 != None:
            entity2_occurences = self.entity_universes[entity_2]
            embedding_space_ids = embedding_space_ids.intersection(entity2_occurences)
        return embedding_space_ids

    def calc_tuple_score(self, local_ent_id, local_rel_id, mode, embedding_space):
        rel_embedding = embedding_space.rel_embeddings(to_tensor(local_rel_id, use_gpu=self.use_gpu))
        ent = embedding_space.ent_embeddings(to_tensor(local_ent_id, use_gpu=self.use_gpu))
        zero_vec = to_tensor([0.0], use_gpu=self.use_gpu)
        if mode == 'head_batch':
            head_embedding = zero_vec
            tail_embedding = ent
        elif mode == 'tail_batch':
            head_embedding = ent
            tail_embedding = zero_vec
        return embedding_space._calc(head_embedding, tail_embedding, rel_embedding, mode)

    def predict_tuple(self, ent_id, rel_id, mode):
        # Calculate f(h,r) if mode == "tail_batch" else Calculate f(r,t)
        embedding_space_ids = self.gather_embedding_spaces(ent_id, rel_id)

        # If incremental is to deprecate spaces of deleted triplpes then exclude them from global energy estimation
        if self.training_setting == "incremental" and self.incremental_strategy == "deprecate":
            embedding_space_ids = embedding_space_ids - self.deprecated_embeddingspaces

        tuple_score = float("inf")
        for embedding_space_id in embedding_space_ids:
            embedding_space = self.trained_embedding_spaces[embedding_space_id]

            local_head_id = self.entity_id_mappings[embedding_space_id][ent_id]
            local_head_id = to_tensor(local_head_id, self.use_gpu)
            local_rel_id = self.relation_id_mappings[embedding_space_id][rel_id]
            local_rel_id = to_tensor(local_rel_id, self.use_gpu)

            embedding_space_tuple_score = self.calc_tuple_score(local_head_id, local_rel_id, mode, embedding_space)
            if embedding_space_tuple_score < tuple_score:
                tuple_score = embedding_space_tuple_score

        return tuple_score

    def predict_triple(self, head_id, rel_id, tail_id, mode='normal'):
        # Gather embedding spaces in which the triple is hold
        embedding_space_ids = self.gather_embedding_spaces(head_id, rel_id, tail_id)

        # If incremental is to deprecate spaces of deleted triplpes then exclude them from global energy estimation
        if self.training_setting == "incremental" and self.incremental_strategy == "deprecate":
            embedding_space_ids = embedding_space_ids - self.deprecated_embeddingspaces

        # Iterate through spaces and get collect the max energy score
        min_energy_score = float("inf")
        for embedding_space_id in embedding_space_ids:
            embedding_space = self.trained_embedding_spaces[embedding_space_id]

            local_head_id = self.entity_id_mappings[embedding_space_id][head_id]
            local_head_id = to_tensor(local_head_id, self.use_gpu)
            local_tail_id = self.entity_id_mappings[embedding_space_id][tail_id]
            local_tail_id = to_tensor(local_tail_id, self.use_gpu)
            local_rel_id = self.relation_id_mappings[embedding_space_id][rel_id]
            local_rel_id = to_tensor(local_rel_id, self.use_gpu)

            energy_score = embedding_space.predict(
                {"batch_h": to_tensor(local_head_id, use_gpu=self.use_gpu),
                 "batch_t": to_tensor(local_tail_id, use_gpu=self.use_gpu),
                 "batch_r": to_tensor(local_rel_id, use_gpu=self.use_gpu),
                 "mode": mode
                 }
            )

            if energy_score < min_energy_score:
                min_energy_score = energy_score

        return min_energy_score

    def transmit_max_scores(self, data, embedding_space_mapping, scores):
        mode = data['mode']
        eval_rel_id = data['batch_r'][0]

        if mode == 'head_batch':
            eval_entity_id = data['batch_t'][0] if mode == 'head_batch' else data['batch_h'][0]
            score_dict = self.evaluation_tail2head_triple_score_dict

        elif mode == 'tail_batch':
            eval_entity_id = data['batch_t'][0] if mode == 'head_batch' else data['batch_h'][0]
            score_dict = self.evaluation_head2tail_triple_score_dict

        dict_key = get_string_key(eval_entity_id, eval_rel_id)
        global_energy_scores = score_dict.setdefault(dict_key, self.default_scores.copy())

        for global_entity_id, local_entity_id in embedding_space_mapping.items():
            entity_score = scores[local_entity_id].item()

            if entity_score < global_energy_scores[global_entity_id]:
                global_energy_scores[global_entity_id] = entity_score

            # def transmit_max_scores(self, data, embedding_space_mapping, scores):
            #     mode = data['mode']
            #     eval_entity_id = data['batch_t'][0] if mode == 'head_batch' else data['batch_h'][0]
            #     eval_rel_id = data['batch_r'][0]
            #
            #     for global_entity_id, local_entity_id in embedding_space_mapping.items():
            #         if mode == 'head_batch'
            #             dict_key = (global_entity_id, eval_rel_id)
            #             entity = eval_entity_id
            #         elif mode == 'tail_batch':
            #             dict_key = (eval_entity_id, eval_rel_id)
            #             entity = global_entity_id
            #
            #         triple_scores = self.evaluation_triple_score_dict[dict_key]
            #         entity_score = scores[local_entity_id]
            #         # get_string_key(global_entity_id, eval_rel_id, eval_entity_id) if mode == 'head_batch' \
            #         # else get_string_key(eval_entity_id, eval_rel_id, global_entity_id)
            #
            #         if not triple_scores:
            #             triple_scores.append(entity, entity_score)
            #
            #         if not triple_scores:
            #             pairs[key].append(c + ':' + freq if freq != '1' else c)
            #
            #         elif entity_score.item() < BinSearch(triple_scores, entity):
            #             self.evaluation_triple_score_dict[min_dict_key] = entity_score.item()

    def transmit_tuple_max_score(self, data, universe_id):
        mode = data["mode"]
        eval_rel_id = data['batch_r'][0]

        if mode == 'head_batch':
            eval_entity_id = data['batch_t'][0] if mode == 'head_batch' else data['batch_h'][0]
            score_dict = self.evaluation_tail2rel_tuple_score_dict

        elif mode == 'tail_batch':
            eval_entity_id = data['batch_t'][0] if mode == 'head_batch' else data['batch_h'][0]
            score_dict = self.evaluation_head2rel_tuple_score_dict

        dict_key = get_string_key(eval_entity_id, eval_rel_id)

        embedding_space = self.trained_embedding_spaces[universe_id]
        local_entity_id = self.entity_id_mappings[universe_id][eval_entity_id]
        local_relation_id = self.relation_id_mappings[universe_id][eval_rel_id]

        local_tuple_score = self.calc_tuple_score(local_entity_id, local_relation_id, mode, embedding_space)
        if local_tuple_score < score_dict.get(dict_key, float_default()):
            score_dict[dict_key] = local_tuple_score

    def obtain_embedding_space_score(self, data, universe_id):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']

        embedding_space_mapping = self.entity_id_mappings[universe_id]
        embedding_space = self.trained_embedding_spaces[universe_id]

        local_batch_h = embedding_space_mapping.keys() if mode == "head_batch" else batch_h
        local_batch_h = [self.entity_id_mappings[universe_id][global_entity_id] for global_entity_id in
                         local_batch_h]
        local_batch_t = batch_t if mode == "head_batch" else embedding_space_mapping.keys()
        local_batch_t = [self.entity_id_mappings[universe_id][global_entity_id] for global_entity_id in
                         local_batch_t]
        local_batch_r = [self.relation_id_mappings[universe_id][global_relation_id] for global_relation_id in
                         batch_r]

        embedding_space_scores = embedding_space.predict(
            {"batch_h": to_tensor(local_batch_h, use_gpu=self.use_gpu),
             "batch_t": to_tensor(local_batch_t, use_gpu=self.use_gpu),
             "batch_r": to_tensor(local_batch_r, use_gpu=self.use_gpu),
             "mode": mode
             }
        )
        # iterate through dict and score tensor (index of both is equal) and transmit scores with comparison to energy_scores
        self.transmit_max_scores(data, embedding_space_mapping, embedding_space_scores)
        self.transmit_tuple_max_score(data, universe_id)

    def reset_evaluation_helpers(self):
        self.current_validated_universes = 0
        self.current_tested_universes = 0

        self.evaluation_head2tail_triple_score_dict.clear()
        self.evaluation_tail2head_triple_score_dict.clear()
        self.evaluation_head2rel_tuple_score_dict.clear()
        self.evaluation_tail2rel_tuple_score_dict.clear()

        self.incremental_strategy = "normal"

    def eval_universes(self, eval_mode):
        # Dependent on mode load validation or test data
        eval_dataloader = self.data_loader if eval_mode == 'test' else self.valid_dataloader
        evaluation_range = tqdm(eval_dataloader)

        # Set range to obtain local energy scores from
        current_evaluated_universes = self.current_tested_universes if eval_mode == 'test' else self.current_validated_universes
        eval_embeddingspaces = [i for i in range(current_evaluated_universes, self.next_universe_id)]

        # If strategy is "deprecate", deprecate embedding spaces in which deleted triples occur by restricting
        # evaluation range
        if self.incremental_strategy == "deprecate":
            self.determine_deprecated_embedding_spaces()
            eval_embeddingspaces = [embedding_space for embedding_space in range(0, self.next_universe_id)
                                    if embedding_space not in self.deprecated_embeddingspaces]

        print("Global energy estimation.")
        print("- Mode: {}".format(eval_mode))
        print("- Training type: {}".format(self.training_setting))
        print("- Incremental optimization strategy : {}".format(self.incremental_strategy))

        if self.incremental_strategy == "deprecate":
            num_deprecated_spaces = len(self.deprecated_embeddingspaces)
            print("-- Deprecated universes: {}".format(num_deprecated_spaces))
            print("-- ... in proportion to the total amount of universes: {}%"
                  .format(num_deprecated_spaces / self.next_universe_id * 100))

        if eval_embeddingspaces:
            print("- Universe range to obtain local energies: ({} -> {})".format(min(eval_embeddingspaces),
                                                                                 max(eval_embeddingspaces)))
            for index, [data_head, data_tail] in enumerate(evaluation_range):
                head = data_tail['batch_h'][0]
                rel = data_head['batch_r'][0]
                tail = data_head['batch_t'][0]

                for universe_id in eval_embeddingspaces:
                    if (universe_id in self.entity_universes[head]) and (universe_id in self.relation_universes[rel]):
                        self.obtain_embedding_space_score(data_tail, universe_id)

                    if (universe_id in self.entity_universes[tail]) and (universe_id in self.relation_universes[rel]):
                        self.obtain_embedding_space_score(data_head, universe_id)

            if eval_mode == 'test':
                self.current_tested_universes = self.next_universe_id
            elif eval_mode == 'valid':
                self.current_validated_universes = self.next_universe_id
        else:
            print("- No universes to be evaluated.")

    def global_energy_estimation(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        eval_rel_id = batch_r[0]
        mode = data['mode']

        if mode == 'head_batch':
            eval_entity_id = batch_t[0]
            evaluation_entities = batch_h
            num_of_evaluation_entitites = len(batch_h)
            score_dict = self.evaluation_tail2head_triple_score_dict

        elif mode == 'tail_batch':
            eval_entity_id = batch_h[0]
            evaluation_entities = batch_t
            num_of_evaluation_entitites = len(batch_t)  # For incremental Setting num of currently contained entities
            score_dict = self.evaluation_head2tail_triple_score_dict

        batch_scores = np.zeros(shape=num_of_evaluation_entitites, dtype=np.float32)
        dict_key = get_string_key(eval_entity_id, eval_rel_id)
        score_dict = score_dict.get(dict_key, [])

        if not score_dict:
            batch_scores[:] = float_default()
        else:
            for idx, entity in enumerate(evaluation_entities):
                batch_scores[idx] = score_dict[entity]

        if self.missing_embedding_handling == 'null_vector':
            tuple_score_dict = self.evaluation_tail2rel_tuple_score_dict if mode == 'head_batch' \
                else self.evaluation_head2rel_tuple_score_dict
            missing_value_replacement = tuple_score_dict.get(dict_key, float_default())

            if missing_value_replacement != float_default():
                batch_scores[batch_scores == float_default()] = missing_value_replacement.item()

        return batch_scores

    def global_energy_estimation2(self, data):
        batch_h = data['batch_h'].numpy() if type(data['batch_h']) == torch.Tensor else data['batch_h']
        batch_t = data['batch_t'].numpy() if type(data['batch_t']) == torch.Tensor else data['batch_t']
        batch_r = data['batch_r'].numpy() if type(data['batch_r']) == torch.Tensor else data['batch_r']
        mode = data['mode']

        triple_entity = batch_t[0] if mode == 'head_batch' else batch_h[0]
        triple_relation = batch_r[0]
        evaluation_entities = batch_h if mode == 'head_batch' else batch_t

        # Gather embedding spaces in which the tuple (entity, relation) is hold
        embedding_space_ids = self.gather_embedding_spaces(triple_entity, triple_relation)

        energy_scores_dict = defaultdict(float)
        default_value = float("inf")
        for entity in evaluation_entities:
            energy_scores_dict[entity] = default_value
        tuple_score = default_value

        for embedding_space_id in embedding_space_ids:
            # Get list with entities which are embedded in this space
            embedding_space_entity_dict = self.entity_id_mappings[embedding_space_id]
            # Calculate scores with embedding_space.predict({batch_h,batch_r,batch_t, mode})
            embedding_space = self.trained_embedding_spaces[embedding_space_id]

            local_batch_h = embedding_space_entity_dict.keys() if mode == "head_batch" else batch_h
            local_batch_h = [self.entity_id_mappings[embedding_space_id][global_entity_id] for global_entity_id in
                             local_batch_h]
            local_batch_t = batch_t if mode == "head_batch" else embedding_space_entity_dict.keys()
            local_batch_t = [self.entity_id_mappings[embedding_space_id][global_entity_id] for global_entity_id in
                             local_batch_t]

            local_batch_r = [self.relation_id_mappings[embedding_space_id][global_relation_id] for global_relation_id in
                             batch_r]

            embedding_space_scores = embedding_space.predict(
                {"batch_h": to_tensor(local_batch_h, use_gpu=self.use_gpu),
                 "batch_t": to_tensor(local_batch_t, use_gpu=self.use_gpu),
                 "batch_r": to_tensor(local_batch_r, use_gpu=self.use_gpu),
                 "mode": mode
                 }
            )
            # iterate through dict and score tensor (index of both is equal) and transmit scores with comparison to energy_scores
            for global_entity_id, local_entity_id in embedding_space_entity_dict.items():
                entity_score = embedding_space_scores[local_entity_id]
                if entity_score < energy_scores_dict[global_entity_id]:
                    energy_scores_dict[global_entity_id] = entity_score

            if self.missing_embedding_handling == 'null_vector':
                local_batch_ent = local_batch_t if mode == "head_batch" else local_batch_h
                local_tuple_score = self.calc_tuple_score(local_batch_ent, local_batch_r, mode, embedding_space)
                if local_tuple_score < tuple_score:
                    tuple_score = local_tuple_score

        scores = np.fromiter(energy_scores_dict.values(), dtype=np.float32)
        if self.missing_embedding_handling == 'null_vector' and tuple_score != default_value:
            scores[scores == default_value] = tuple_score

        return scores

    def test_one_step(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']

        score = None
        if mode == 'head_batch' or mode == 'tail_batch':
            score = self.global_energy_estimation(data)

        elif mode == 'normal':
            num_of_scores = batch_h.size
            score = np.zeros(shape=num_of_scores, dtype=np.float32)
            for index in range(num_of_scores):
                head, tail, rel = batch_h[index], batch_r[index], batch_t[index]
                global_energy_score = self.predict_triple(head, rel, tail, mode)
                # In case no embedding space could be found for the triple we calculate two tuple scores,
                # i.e. (h,r) and (r,t) and use the higher score of them to discriminate missing values
                if global_energy_score == float("inf") and self.missing_embedding_handling == "null_vector":
                    tuple_score_head_rel = self.predict_tuple(head, rel, mode="tail_batch")
                    tuple_score_rel_tail = self.predict_tuple(tail, rel, mode="head_batch")

                    if tuple_score_head_rel < tuple_score_rel_tail:
                        global_energy_score = tuple_score_head_rel
                    else:
                        global_energy_score = tuple_score_rel_tail

                score[index] = global_energy_score

        return score

    def run_link_prediction(self, type_constrain=False):
        self.data_loader.set_sampling_mode('link')
        self.eval_universes(eval_mode='test')
        mrr, mr, hit10, hit3, hit1 = super().run_link_prediction(type_constrain)
        print('Mean Reciprocal Rank: {}'.format(mrr))
        print('Mean Rank: {}'.format(mr))
        print('Hits@10: {}'.format(hit10))
        print('Hits@3: {}'.format(hit3))
        print('Hits@1: {}'.format(hit1))

    def run_triple_classification(self, threshlod=None):
        # self.eval_universes(eval_mode='test')
        acc, threshlod = super().run_triple_classification(threshlod)
        print("Accuracy is: {}".format(acc))
        return acc, threshlod

    # Adapter which returns negative examples in a datastructure which is compatible with the triplce classification
    # method of the OpenKE framework
    def tc_datastructure_adapter(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
        return [({
                     'batch_h': np.asarray(pos_h, dtype=np.int64) if pos_h else np.empty(0, dtype=np.int64),
                     'batch_t': np.asarray(pos_t, dtype=np.int64) if pos_t else np.empty(0, dtype=np.int64),
                     'batch_r': np.asarray(pos_r, dtype=np.int64) if pos_r else np.empty(0, dtype=np.int64),
                     "mode": "normal"
                 },
                 {
                     'batch_h': np.asarray(neg_h, dtype=np.int64) if neg_h else np.empty(0, dtype=np.int64),
                     'batch_t': np.asarray(neg_t, dtype=np.int64) if neg_t else np.empty(0, dtype=np.int64),
                     'batch_r': np.asarray(neg_r, dtype=np.int64) if neg_r else np.empty(0, dtype=np.int64),
                     "mode": "normal"
                 })]

    def load_triple_classification_file(self, file):
        pos_h = []
        pos_t = []
        pos_r = []
        neg_h = []
        neg_t = []
        neg_r = []

        with file.open(mode="rt", encoding="UTF-8") as f:
            for line in f:
                head, tail, rel, truth_value = line.split()

                if truth_value == "1":
                    pos_h.append(int(head))
                    pos_t.append(int(tail))
                    pos_r.append(int(rel))

                elif truth_value == "0":
                    neg_h.append(int(head))
                    neg_t.append(int(tail))
                    neg_r.append(int(rel))

        return pos_h, pos_t, pos_r, neg_h, neg_t, neg_r

    def run_classification_of_deleted_triples(self, snapshot_idx, threshlod):
        # Load negative examples of snapshot
        snapshot_folder = Path(self.data_loader.in_path) / "incremental" / str(snapshot_idx)
        negative_example_files = [file for file in snapshot_folder.iterdir() if file.name.startswith("tc_")]
        for file in negative_example_files:
            pos_h, pos_t, pos_r, neg_h, neg_t, neg_r = self.load_triple_classification_file(file)

            tc_data = self.tc_datastructure_adapter(pos_h, pos_t, pos_r, neg_h, neg_t, neg_r)
            acc, _ = super().run_triple_classification(threshlod, data_iterator=tc_data)
            print("Accuracy for {} is: {}".format(file.name, acc))

    def run_triple_classification_from_files(self, snapshot):
        # Load basic test examples of snapshot
        snapshot_folder = Path(self.data_loader.in_path) / "incremental" / str(snapshot)
        triple_classification_file = snapshot_folder / "triple_classification_prepared_test_examples.txt"

        pos_h, pos_t, pos_r, neg_h, neg_t, neg_r = self.load_triple_classification_file(triple_classification_file)

        tc_data = self.tc_datastructure_adapter(pos_h, pos_t, pos_r, neg_h, neg_t, neg_r)
        acc, threshlod = super().run_triple_classification(data_iterator=tc_data)
        print("Accuracy for {} is: {}".format(triple_classification_file.name, acc))
        print("Determined threshold: {}".format(threshlod))

        print("Run negative triple classification...")
        self.run_classification_of_deleted_triples(snapshot, threshlod)

    def determine_deprecated_embedding_spaces(self):
        if self.deprecated_embeddingspaces:
            self.deprecated_embeddingspaces.clear()
        for triple in self.train_dataloader.deleted_triple_set:
            head, tail, rel = triple
            embedding_space_ids_set = self.gather_embedding_spaces(int(head), int(rel), int(tail))
            self.deprecated_embeddingspaces.update(embedding_space_ids_set)

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
                ParallelUniverse_inst.relation_universes[relation])  # relation_id -> universe_id

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
        state_dict = {'initial_num_universes': self.initial_num_universes,
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

                      'min_balance': self.min_balance,
                      'max_balance': self.max_balance,

                      'embedding_model': self.embedding_model,
                      'embedding_model_param': self.embedding_model_param,

                      'best_hit10': self.best_hit10,
                      'bad_counts': self.bad_counts,
                      }

        if self.training_setting == "static":
            eval_universes_dict = {'current_tested_universes': self.current_tested_universes,
                                   'current_validated_universes': self.current_validated_universes,
                                   'evaluation_head2tail_triple_score_dict': self.evaluation_head2tail_triple_score_dict,
                                   'evaluation_tail2head_triple_score_dict': self.evaluation_tail2head_triple_score_dict,
                                   'evaluation_head2rel_tuple_score_dict': self.evaluation_head2rel_tuple_score_dict,
                                   'evaluation_tail2rel_tuple_score_dict': self.evaluation_tail2rel_tuple_score_dict}
            state_dict.update(eval_universes_dict)

        return state_dict

    def process_state_dict(self, state_dict):
        self.initial_num_universes = state_dict['initial_num_universes']
        self.next_universe_id = state_dict['next_universe_id']
        self.trained_embedding_spaces = state_dict['trained_embedding_spaces']
        self.entity_id_mappings = state_dict['entity_id_mappings']
        self.relation_id_mappings = state_dict['relation_id_mappings']
        self.entity_universes = state_dict['entity_universes']
        self.relation_universes = state_dict['relation_universes']
        self.min_margin = state_dict['min_margin']
        self.max_margin = state_dict['max_margin']
        self.min_lr = state_dict['min_lr']
        self.max_lr = state_dict['max_lr']
        self.min_num_epochs = state_dict['min_num_epochs']
        self.max_num_epochs = state_dict['max_num_epochs']
        self.min_triple_constraint = state_dict['min_triple_constraint']
        self.max_triple_constraint = state_dict['max_triple_constraint']
        if 'min_balance' in state_dict:
            self.min_balance = state_dict['min_balance']
            self.max_balance = state_dict['max_balance']
        # if 'num_dim' in state_dict:
        #     self.num_dim = state_dict['num_dim']
        #     self.norm = state_dict['p_norm']
        # if 'embedding_method' in state_dict:
        #     self.embedding_method = state_dict['embedding_method']
        elif 'embedding_model' in state_dict:
            self.embedding_model = state_dict['embedding_model']
            self.embedding_model_param = state_dict['embedding_model_param']
        if 'best_hit10' in state_dict:
            self.best_hit10 = state_dict['best_hit10']
            self.bad_counts = state_dict['bad_counts']

        if '' in state_dict:
            self.current_tested_universes = state_dict['current_tested_universes']
            self.current_validated_universes = state_dict['current_validated_universes']
            self.evaluation_head2tail_triple_score_dict = state_dict['evaluation_head2tail_triple_score_dict']
            self.evaluation_tail2head_triple_score_dict = state_dict['evaluation_tail2head_triple_score_dict']
            self.evaluation_head2rel_tuple_score_dict = state_dict['evaluation_head2rel_tuple_score_dict']
            self.evaluation_tail2rel_tuple_score_dict = state_dict['evaluation_tail2rel_tuple_score_dict']

    def save_parameters(self, path):
        state_dict = self.extend_state_dict()
        torch.save(state_dict, path)

    def load_parameters(self, filename):
        state_dict = torch.load(self.checkpoint_dir + filename)
        self.process_state_dict(state_dict)

    def calculate_unembedded_ratio(self, mode='examine_entities'):
        num_unembedded = 0
        mapping_dict = self.entity_universes if mode == 'examine_entities' else self.relation_universes
        num_total = self.train_dataloader.entTotal if mode == 'examine_entities' else self.train_dataloader.relTotal

        for i in range(num_total):
            if len(mapping_dict[i]) == 0:
                num_unembedded += 1

        return num_unembedded / num_total
