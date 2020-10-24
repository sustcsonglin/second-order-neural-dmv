# -*- coding: utf-8 -*-

from parser.cmds.cmd import CMD


class Predict(CMD):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Use a trained model to make predictions.'
        )

        return subparser

    def __call__(self, args):
        super(Predict, self).__call__(args)
