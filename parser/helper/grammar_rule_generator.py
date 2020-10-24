from parser.helper.grammar_rule_padder import *
from parser.helper.generate_rules_from_results import *


class RuleGenerator():
    def get_child_rule(self):
        raise NotImplementedError

    def get_decision_rule(self):
        raise NotImplementedError

    def get_root_rule(self):
        return generate_root_rule

    def pad_child_rule(self):
        raise NotImplementedError

    def pad_decision_rule(self):
        raise NotImplementedError

    def pad_root_rule(self):
        return RootRulePadder()

class RuleGenerator1o(RuleGenerator):

    def pad_child_rule(self):
        return ChildRulePadder_1o()

    def pad_decision_rule(self):
        return DecisionRulePadder_1o()

    def get_child_rule(self):
        return generate_attach_rule_1o

    def get_decision_rule(self):
        return generate_decision_rule_1o

class RuleGeneratorSib(RuleGenerator):
    def pad_child_rule(self):
        return ChildRulePadderSib()

    def pad_decision_rule(self):
        return DecisionRulePadderSib()

    def get_child_rule(self):
        return generate_attach_rule_2o_sib

    def get_decision_rule(self):
        return generate_decision_rule_2o_sib


