import numpy as np
from parser.const import *


def generate_unlabeled_tree_mask(heads):
    seq_len = len(heads)
    gold = np.zeros(shape=(seq_len+1, seq_len+1),dtype=np.float )
    gold.fill(-1e9)
    for child, head in enumerate(heads):
        gold[head][child + 1] = 0
    return gold


def generate_decision_rule_1o(heads):
    '''
    First-order DMV, generate the grammar rules used in the "predicted" parse tree from other parser.
    :param heads: the head of each position
    :return: decision rule
    '''
    seq_len = len(heads)
    decision = np.zeros(shape=(seq_len, 2, 2, 2))
    left_most_child = list(range(seq_len))
    right_most_child = list(range(seq_len))
    for child, head in enumerate(heads):
        head = head - 1
        if head == -1:
            continue
        elif child < head:
            if child < left_most_child[head]:
                left_most_child[head] = child
        else:
            if child > right_most_child[head]:
                right_most_child[head] = child

    for child, head in enumerate(heads):
        head = head - 1
        if child < head:
            if left_most_child[head] != child:
                decision[head][LEFT][HASCHILD][GO] += 1
            else:
                decision[head][LEFT][NOCHILD][GO] += 1
        else:
            if right_most_child[head] != child:
                decision[head][RIGHT][HASCHILD][GO] += 1
            else:
                decision[head][RIGHT][NOCHILD][GO] += 1

        if left_most_child[child] == child:
            decision[child][LEFT][NOCHILD][STOP] += 1
        else:
            decision[child][LEFT][HASCHILD][STOP]+= 1

        if right_most_child[child] == child:
            decision[child][RIGHT][NOCHILD][STOP] += 1
        else:
            decision[child][RIGHT][HASCHILD][STOP] += 1

    return decision


def generate_attach_rule_1o(heads):
    seq_len = len(heads)
    attach = np.zeros(shape=(seq_len, seq_len, 2))
    left_most_child = list(range(seq_len))
    right_most_child = list(range(seq_len))
    for child, head in enumerate(heads):
        head = head - 1
        if head == -1:
            continue
        elif child < head:
            if child < left_most_child[head]:
                left_most_child[head] = child
        else:
            if child > right_most_child[head]:
                right_most_child[head] = child

    for child, head in enumerate(heads):
        head = head - 1
        if head == -1:
            continue
        if child < head:
            if left_most_child[head] != child:
                attach[head][child][HASCHILD] += 1
            else:
                attach[head][child][NOCHILD] += 1
        else:
            if right_most_child[head] != child:
                attach[head][child][HASCHILD] += 1
            else:
                attach[head][child][NOCHILD] += 1
    return attach

def generate_root_rule(heads):
    seq_len = len(heads)
    root = np.zeros(shape=(seq_len,))
    for child, head in enumerate(heads):
        head = head - 1
        if head == -1:
            root[child] = 1
            return root


def generate_attach_rule_2o_sib(heads):
    '''
    Default generation order: generate the furthest child first.
    sibling 0 means no child is further than it in given direction
    example:
    for a given sentence (1, 2, 3, 4, 5)
    if we have following arcs:
    (3, 2)  (3, 1)   Head 3 generate 1 first, with sibling 0 (no sib), valence NOCHILD. then head 3 generate 2, with sibling 1, valence HASCHILD.
    (3, 5)  (3, 4)   Head 3 generate 5 first, with sibling 0 (no sib), valence NOCHILD. then head 3 generate 4, with sibling 5, valence HASCHILD.
    '''
    seq_len = len(heads)
    attach = np.zeros(shape=(seq_len, seq_len, seq_len+1, 2))
    left_most_child = list(range(seq_len))
    right_most_child = list(range(seq_len))
    for child, head in enumerate(heads):
        head = head - 1
        if head == -1:
            continue
        elif child < head:
            if child < left_most_child[head]:
                left_most_child[head] = child
        else:
            if child > right_most_child[head]:
                right_most_child[head] = child

    for child, head in enumerate(heads):
        head = head - 1
        if head == -1:
            continue
        if child < head:
            prev_child = -1
            for prev in range(0, child):
                if heads[prev] - 1 == head:
                    prev_child = prev
            sibling = prev_child + 1
            if left_most_child[head] != child:
                attach[head][child][sibling][HASCHILD] += 1
            else:
                attach[head][child][sibling][NOCHILD] += 1
        else:
            prev_child = -1
            for prev in range(child+1, len(heads)):
                if heads[prev] - 1 == head:
                    prev_child = prev
                    break
            sibling = prev_child + 1
            if right_most_child[head] != child:
                attach[head][child][sibling][HASCHILD] += 1
            else:
                attach[head][child][sibling][NOCHILD] += 1
    return attach



def generate_decision_rule_2o_sib(heads):
    '''
    Default generation order: generate the furthest child first.
    0th sibling  means no child is further than it in given direction
    example:
    for a given sentence (1, 2, 3, 4, 5)
    if we have following arcs:  (3, 2)  (3, 1)
    The generative story comes:
    Head 3 generate 1 first, here the decision rule is "head3 generate a child in the LEFT hand side,
    valence: NOCHILD, sibling:0, decision: GO) (LEFT, NOCHILD, sibling=0, GO)
     Then, Head 3 generate 2, here the decision rule is "head3 generate a child in the LEFT hand size, valence: HASCHILD,
     sibling: 1, decision: GO)
    Finally, Head 3 has no further child in the left hand side. Since it has already generated children, the decision rule is
    (LEFT, HASCHILD, sibling=1, STOP), sibling is the last child it generated in this direction.
    '''

    seq_len = len(heads)
    decision = np.zeros(shape=(seq_len, seq_len+1, 2, 2, 2))
    left_most_child = list(range(seq_len))
    right_most_child = list(range(seq_len))
    left_inner_most_child = [-1] * seq_len
    right_inner_most_child = [-1] * seq_len
    for child, head in enumerate(heads):
        head = head - 1
        if head == -1:
            continue
        elif child < head:
            if child < left_most_child[head]:
                left_most_child[head] = child
            left_inner_most_child[head] = child
        else:
            if child > right_most_child[head]:
                right_most_child[head] = child
            if right_inner_most_child[head] == -1:
                right_inner_most_child[head] = child
    for child, head in enumerate(heads):
        head = head - 1
        if head == -1:
            continue
        if child < head:
            prev_child = -1
            for prev in range(0, child):
                if heads[prev] - 1 == head:
                    prev_child = prev
            if left_most_child[head] != child:
                decision[head][prev_child + 1][LEFT][HASCHILD][GO] += 1
            else:
                decision[head][prev_child + 1][LEFT][NOCHILD][GO] += 1
        else:
            prev_child = -1
            for prev in range(child+1, len(heads)):
                if heads[prev] - 1 == head:
                    prev_child = prev
                    break
            if right_most_child[head] != child:
                decision[head][prev_child+1][RIGHT][HASCHILD][GO] += 1
            else:
                decision[head][prev_child+1][RIGHT][NOCHILD][GO] += 1
        if left_inner_most_child[child] == -1:
            decision[child][0][LEFT][NOCHILD][STOP] += 1
        else:
            decision[child][left_inner_most_child[child]+1][LEFT][HASCHILD][STOP] += 1
        if right_inner_most_child[child] == -1:
            decision[child][0][RIGHT][NOCHILD][STOP] += 1
        else:
            decision[child][right_inner_most_child[child]+1][RIGHT][HASCHILD][STOP] += 1
    return decision



def generate_decision_rule_2o_grand(heads):
    pass



