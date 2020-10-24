
from parser.helper.grammar_rule_generator import RuleGenerator1o, RuleGeneratorSib
import torch
from parser.const import *

from fastNLP.io.loader.conll import ConllLoader
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.core.batch import  DataSetIter
from fastNLP.core.sampler import  ConstantTokenNumSampler, RandomSampler, BucketSampler
from gensim.models import FastText
import numpy as np


class DataModule():
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.device = self.hparams.device
        self.setup(hparams.mode)


    def prepare_data(self):
        #download data from
        pass




    def setup(self, stage="train"):
        if stage == 'train':
            data = self.hparams.data
            # build dataset
            # indexes: the ith column of the conll file. it depends on your file and may need modification.
            loader = ConllLoader([word, pos, head], indexes=[1, 3, 6])
            train_dataset = loader._load(data.train_file)
            val_dataset = loader._load(data.val_file)
            test_dataset = loader._load(data.test_file)

            def clean_word(words):
                def clean_number(word):
                    def is_number(s):
                        try:
                            float(s)
                            return True
                        except ValueError:
                            return False
                    if is_number(word):
                        return '0'
                    else:
                        return word
                # import re
                # def clean_number(w):
                #     new_w = re.sub('[0-9]{1,}([,.]?[0-9]*)*', '0', w)
                #     return new_w
                return [clean_number(word) for word in words]

            def numerize(heads):
                return [int(head) for head in heads]

            train_dataset.apply_field(clean_word, word, new_field_name=word)
            val_dataset.apply_field(clean_word, word, new_field_name=word)
            test_dataset.apply_field(clean_word, word, new_field_name=word)
            test_dataset.apply_field(numerize, head, new_field_name=head)
            train_dataset.add_seq_len(field_name=word, new_field_name=seq_len)
            val_dataset.add_seq_len(field_name=word, new_field_name=seq_len)
            test_dataset.add_seq_len(field_name=word, new_field_name=seq_len)


            pos_vocab = Vocabulary()
            pos_vocab.from_dataset(train_dataset, field_name=pos)

            if data.wordposastoken:
                '''
                combining pos tag and word as a single token.
                Largely speaking, we build the vocabulary based on the co-occurance of (NT, 'word')
                Then, we replace all unknown word with their corresponding POS tag.
                Please refer
                "Dependency Grammar Induction with Neural Lexicalization and Big Training Data"
                for details.
                '''
                def combine(x):
                    sent = list(zip(x[pos], x[word]))
                    return [x[0] + "_" + x[1] for x in sent]

                train_dataset.apply(combine, new_field_name=word)
                val_dataset.apply(combine, new_field_name=word)
                test_dataset.apply(combine, new_field_name=word)
                word_vocab = Vocabulary(min_freq=data.min_freq)
                word_vocab.from_dataset(train_dataset, field_name=word)

                '''
                Replace the unknown word with their POS tag.
                '''

                word_vocab.add_word_lst(pos_vocab.word2idx)
                word_vocab.index_dataset(train_dataset, field_name=word)
                word_vocab.index_dataset(val_dataset, field_name=word)
                word_vocab.index_dataset(test_dataset, field_name=word)
                unk = 1

                def replace(x):
                    poses = x[pos]
                    words = x[word]
                    for i in range(len(words)):
                        # 1 stands for unk. we replace the unknown word with its POS tags.
                        if words[i] == unk:
                            pos_tag_name = poses[i]
                            words[i] = word_vocab[pos_tag_name]
                    return words

                train_dataset.apply(replace, new_field_name=word)
                val_dataset.apply(replace, new_field_name=word)
                test_dataset.apply(replace, new_field_name=word)

                if data.use_emb:
                    if data.emb_type == 'fasttext':
                        model = FastText.load(data.embedding)
                    else:
                        raise NotImplementedError
                    word_vec = model.wv
                    emb = np.random.rand(len(word_vocab), data.word_emb_size)
                    for idx, w in word_vocab.idx2word.items():
                        if "_" in w:
                            w = w.split('_')[-1]
                            emb[idx] = word_vec[w]
                    emb = torch.from_numpy(emb)
                    self.pretrained_emb = emb.to(self.device).float()

                word2pos = np.zeros(shape=(len(word_vocab),))

                # to match each token in vocabulary with its corresponding POS tag.
                for idx, w in word_vocab.idx2word.items():
                    if idx == 0:
                        continue
                    if idx == 1:
                        word2pos[1] = 1
                        continue
                    if "_" in w:
                        pos_tag_name = w.split("_")[0]
                        word2pos[idx] = pos_vocab.word2idx[pos_tag_name]
                    else:
                        word2pos[idx] = pos_vocab.word2idx[w]
                self.word2pos = torch.from_numpy(word2pos).long().to(self.device)


            # if not combine pos/word as a single token.
            else:
                # choose the create the vocabulary with fix size or based on the word frequency.
                if data.vocab_type == 'max_size':
                    word_vocab = Vocabulary(max_size=data.vocab_size)
                else:
                    word_vocab = Vocabulary(min_freq=data.min_freq)
                word_vocab.from_dataset(train_dataset, field_name=word)
                word_vocab.index_dataset(train_dataset, field_name=word)
                word_vocab.index_dataset(val_dataset, field_name=word)
                word_vocab.index_dataset(test_dataset, field_name=word)

            train_dataset.set_input(pos, word, seq_len)
            val_dataset.set_input(pos, word, seq_len)
            test_dataset.set_input(pos, word, seq_len)
            test_dataset.set_target(head)

            pos_vocab.index_dataset(train_dataset, field_name=pos)
            pos_vocab.index_dataset(val_dataset, field_name=pos)
            pos_vocab.index_dataset(test_dataset, field_name=pos)

            train_dataset_init = None

            '''
            Use external unsupervised parser's parse result as "psudo-gold-tree" to initialize our model.
            '''
            if self.hparams.train.initializer == 'external':
                # dependent on your file format.
                conll_loader = ConllLoader([word, pos, head], indexes=[1, 4, 6])
                train_dataset_init = conll_loader._load(data.external_parser)
                train_dataset_init.add_seq_len(field_name=word, new_field_name=seq_len)
                train_dataset_init.apply_field(clean_word, word, new_field_name=word)
                train_dataset_init.apply_field(numerize, head, new_field_name=head)

                if not data.wordposastoken:
                    word_vocab.index_dataset(train_dataset_init, field_name=word)
                else:
                    train_dataset_init.apply(combine, new_field_name=word)
                    word_vocab.index_dataset(train_dataset_init, field_name=word)
                    train_dataset_init.apply(replace, new_field_name=word)

                pos_vocab.index_dataset(train_dataset_init, field_name=pos)

                if self.hparams.joint_training:
                    import copy
                    train_dataset_init_for_model2 = copy.deepcopy(train_dataset_init)

                # first-order model
                if (self.hparams.model.model_name == 'NeuralDMV') or (self.hparams.model.model_name == 'LexicalizedNDMV'):
                    rule_generator = RuleGenerator1o()

                # second-order model
                elif self.hparams.model.model_name == 'SiblingNDMV':
                    rule_generator = RuleGeneratorSib()

                elif self.hparams.model.model_name == 'JointFirstSecond':
                    rule_generator = RuleGenerator1o()
                    rule_generator_for_model2 = RuleGeneratorSib()

                else:
                    raise NameError

                self.setup_init_dataset(train_dataset_init, rule_generator)

                if self.hparams.joint_training:
                    self.setup_init_dataset(train_dataset_init_for_model2, rule_generator_for_model2)


            elif self.hparams.train.initializer == 'km':
                train_dataset_init = train_dataset

            self.pos_vocab = pos_vocab
            self.word_vocab = word_vocab
            self.train_dataset = train_dataset
            self.val_dataset =  val_dataset
            self.test_dataset = test_dataset
            self.train_dataset_init = train_dataset_init
            if self.hparams.joint_training:
                self.train_dataset_init_for_model2 = train_dataset_init_for_model2

        else:
            raise NotImplementedError


    def setup_init_dataset(self, dataset, rule_generator):
        dataset.apply_field(func=rule_generator.get_child_rule(), field_name=head, new_field_name=attach_name)
        dataset.apply_field(func=rule_generator.get_decision_rule(), field_name=head,
                                       new_field_name=decision_name)
        dataset.apply_field(func=rule_generator.get_root_rule(), field_name=head, new_field_name=root_name)
        dataset.set_input(pos, word, seq_len)
        dataset.set_target(attach_name, decision_name, root_name)
        dataset.set_padder(field_name=attach_name, padder=rule_generator.pad_child_rule())
        dataset.set_padder(field_name=decision_name, padder=rule_generator.pad_decision_rule())
        dataset.set_padder(field_name=root_name, padder=rule_generator.pad_root_rule())

    @property
    def train_dataloader(self):
        # Random
        args = self.hparams.train.training
        train_sampler = RandomSampler()
        train_loader = DataSetIter(batch_size=args.batch_size,
                                   dataset=self.train_dataset, sampler=train_sampler, drop_last=False)
        return train_loader

    @property
    def val_dataloader(self):
        args = self.hparams.test
        val_sampler = ConstantTokenNumSampler(seq_len=self.val_dataset.get_field(seq_len).content,
                                              max_token=args.max_tokens, num_bucket=args.bucket)
        val_loader = DataSetIter(self.val_dataset, batch_size=1, sampler=None, as_numpy=False, num_workers=4,
                                  pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                                  batch_sampler=val_sampler)
        return val_loader

    @property
    def test_dataloader(self):
        args = self.hparams.test
        test_sampler = ConstantTokenNumSampler(seq_len=self.test_dataset.get_field(seq_len).content,
                                               max_token=args.max_tokens, num_bucket=args.bucket)
        test_loader = DataSetIter(self.test_dataset, batch_size=1, sampler=None, as_numpy=False, num_workers=4,
                                  pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                                  batch_sampler=test_sampler)
        return test_loader


    @property
    def train_init_dataloader(self):
        if self.train_dataset_init is None:
            return None

        init_sampler = RandomSampler()
        args = self.hparams.train.init if not self.hparams.joint_training else self.hparams.train.init.model1
        init_loader = DataSetIter(batch_size=args.batch_size,
                                   dataset=self.train_dataset_init,
                                   sampler=init_sampler,
                                   drop_last=False)
        return init_loader

    @property
    def train_init_dataloder_for_model2(self):
        assert self.hparams.joint_training
        init_sampler = RandomSampler()
        args = self.hparams.train.init.model2
        init_loader = DataSetIter(batch_size=args.batch_size,
                                  dataset=self.train_dataset_init_for_model2,
                                  sampler=init_sampler,
                                  drop_last=False)
        return init_loader











