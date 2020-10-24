from parser.dmvs.dmv import DMV
import torch
from parser.const import *


'''
Implementation of grand DMV.
'''
class DMV2o_grand(DMV):
    def __init__(self, device):
        super(DMV2o_grand, self).__init__(device)

    def _inside(self, rules, lens, mbr=False, viterbi=False):
        pass