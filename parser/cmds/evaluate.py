# -*- coding: utf-8 -*-

from parser.cmds.cmd import CMD
import torch


class Evaluate(CMD):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Train a model.'
        )

        return subparser



    def __call__(self, args):
        pass