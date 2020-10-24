import numpy as np
import torch
import torch.nn as nn
from parser.model.VanillaDMV import VanillaDMV

class KM_initializer:
    def __init__(self, dataset):
        self.device = dataset.device
        self.d_valence_num = 2
        self.num_state = len(dataset.pos_vocab)
        self.count_smoothing = 0.001
        self.init_model = VanillaDMV(dataset).to(self.device)
        self.fit(dataset.train_dataset)

    def fit(self, dataset):
        def init_param():
            norm_counter = np.zeros((self.num_state, 2, self.d_valence_num, 2))  # shape same as self.decision_param
            self.root_param = np.zeros(shape=(self.num_state,))
            self.transition_param = np.zeros(shape=(self.num_state, self.num_state, 2, self.d_valence_num))
            self.decision_param = np.zeros(shape=(self.num_state, 2, 2, 2))
            for instance in dataset:
                instance = instance['pos_tag']
                word_num = len(instance)  # not include *root*
                change = np.zeros((word_num, 2))
                instance_length = len(instance)
                for entry_idx in range(instance_length):
                    self.root_param[instance[entry_idx]] += 1. / word_num
                for child_i in range(instance_length):
                    child_sum = 0.
                    for head_i in range(instance_length):
                        if child_i == head_i:
                            continue
                        child_sum += 1. / abs(child_i - head_i)
                    if child_sum > 0:
                        scale = (word_num - 1) / word_num / child_sum
                        for head_i in range(instance_length):
                            if child_i == head_i:
                                continue
                            direction = 0 if head_i > child_i else 1
                            head_pos = instance[head_i]
                            child_pos = instance[child_i]
                            diff = scale / abs(head_i - child_i)
                            self.transition_param[head_pos, child_pos, direction, :] += diff
                            change[head_i, direction] += diff
                update_decision(change, norm_counter, instance)
            self.transition_param += self.count_smoothing
            self.decision_param += self.count_smoothing
            self.root_param += self.count_smoothing
            """first_child_update: find min change/count"""
            es = first_child_update(norm_counter)
            norm_counter *= 0.9 * es
            self.decision_param += norm_counter
            root_param_sum = np.sum(self.root_param)
            trans_param_sum = np.sum(self.transition_param, axis=1, keepdims=True)
            decision_param_sum = np.sum(self.decision_param, axis=3, keepdims=True)
            self.root_param /= root_param_sum
            self.transition_param /= trans_param_sum
            self.decision_param /= decision_param_sum
            self.transition_param = torch.tensor(np.log(self.transition_param), dtype=torch.float32, device=self.device)
            self.root_param = torch.tensor(np.log(self.root_param), dtype=torch.float32, device=self.device)
            self.decision_param = torch.tensor(np.log(self.decision_param), dtype=torch.float32, device=self.device)

        def update_decision(change, norm_counter, pos_np):
            # change:         position, direction
            # trans_param:    [given_head_p, given_child_p, direction, :]
            # norm_counter:   pos, direction, dv, decision
            # decision_param: pos, direction, dv, decision
            word_num, _ = change.shape
            for i in range(word_num):
                pos = pos_np[i]
                for direction in range(2):
                    if change[i, direction] > 0:
                        # + and - are just for distinguish, see self.first_child_update
                        norm_counter[pos, direction, 1, 0] += 1
                        norm_counter[pos, direction, 0, 0] += -1
                        self.decision_param[pos, direction, 0, 0] += \
                            change[i, direction]
                        norm_counter[pos, direction, 1, 1] += -1

                        norm_counter[pos, direction, 0, 1] += 1
                        self.decision_param[pos, direction, 1, 1] += 1
                    else:
                        self.decision_param[pos, direction, 1, 1] += 1

        def first_child_update(norm_counter):
            es = 1.0
            all_param = self.decision_param.flatten()
            all_norm = norm_counter.flatten()
            for i in range(len(all_norm)):
                if all_param[i] > 0 > all_norm[i]:
                    ratio = -all_param[i] / all_norm[i]
                    if es > ratio:
                        es = ratio
            return es
        init_param()
        self.init_model._initialize({'attach':self.transition_param, 'decision':self.decision_param, 'root':self.root_param})

    def initialize(self, model, dmv, loader, optimizer, hparams):
        model.train()
        for epoch in range(hparams.max_epoch):
            for x, y in loader:
                optimizer.zero_grad()
                rules = model(x)
                initializer_rule = self.init_model(x)
                grad = dmv._inside(initializer_rule, x['seq_len'], get_expected_counts=True, viterbi=True)
                loss = torch.sum(rules['attach'] * grad['attach_grad'].detach()) + \
                       torch.sum(rules['decision'] * grad['decision_grad'].detach()) + \
                       torch.sum(rules['root'] * grad['root_grad'].detach())
                loss = -loss/rules['attach'].shape[0]
                loss.backward()
                optimizer.step()
            print("Initialization epoch{} finished".format(epoch))
        print("K&M initialization finished.")
















