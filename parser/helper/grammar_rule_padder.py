from fastNLP.core.field import Padder
import numpy as np




class ChildRulePadder_1o(Padder):
    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        max_sent_length = max(rule.shape[0] for rule in contents)
        batch_size = len(contents)
        padded_array = np.full((batch_size, max_sent_length, max_sent_length, 2), fill_value=self.pad_val,
                               dtype=np.float)
        for b_idx, rule in enumerate(contents):
            sent_len = rule.shape[0]
            padded_array[b_idx, :sent_len, :sent_len, :] = rule
        return padded_array

class ChildRulePadderSib(Padder):
    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        max_sent_length = max(rule.shape[0] for rule in contents)
        batch_size = len(contents)
        padded_array = np.full((batch_size, max_sent_length, max_sent_length, max_sent_length+1, 2), fill_value=self.pad_val,
                               dtype=np.float)
        for b_idx, rule in enumerate(contents):
            sent_len = rule.shape[0]
            padded_array[b_idx, :sent_len, :sent_len, :sent_len + 1, :] = rule
        return padded_array


class DecisionRulePadder_1o(Padder):
    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        max_sent_length = max(rule.shape[0] for rule in contents)
        batch_size = len(contents)
        padded_array = np.full((batch_size, max_sent_length, 2, 2, 2), fill_value=self.pad_val,
                               dtype=np.float)
        for b_idx, rule in enumerate(contents):
            sent_len = rule.shape[0]
            padded_array[b_idx, :sent_len] = rule
        return padded_array

class DecisionRulePadderSib(Padder):
    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        max_sent_length = max(rule.shape[0] for rule in contents)
        batch_size = len(contents)
        padded_array = np.full((batch_size, max_sent_length, max_sent_length+1,  2, 2, 2), fill_value=self.pad_val,
                               dtype=np.float)
        for b_idx, rule in enumerate(contents):
            sent_len = rule.shape[0]
            padded_array[b_idx, :sent_len, :sent_len+1] = rule
        return padded_array

class RootRulePadder(Padder):
    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        max_sent_length = max(rule.shape[0] for rule in contents)
        batch_size = len(contents)
        padded_array = np.full((batch_size, max_sent_length, ), fill_value=self.pad_val,
                               dtype=np.float)
        for b_idx, rule in enumerate(contents):
            sent_len = rule.shape[0]
            padded_array[b_idx, :sent_len] = rule
        return padded_array

