import time
import os
import logging
from distutils.dir_util import copy_tree
from parser.model import LVDMV, JointFirstSecondOrderModel, LNDMV, SiblingNDMV, NDMV
from parser.dmvs import DMV1o, DMV2o_sib, DMV_lv

models = {'NeuralDMV': NDMV,
          'SiblingNDMV': SiblingNDMV,
          'LexicalizedNDMV': LNDMV,
          'JointFirstSecond': JointFirstSecondOrderModel,
          'LatentVariableDMV':  LVDMV,
          }

dmvs = {'NeuralDMV': DMV1o,
          'SiblingNDMV': DMV2o_sib,
          'LexicalizedNDMV': DMV1o,
          'JointFirstSecond': DMV2o_sib,
           'LatentVariableDMV': DMV_lv
          }


def get_model(args, dataset):
    model = models[args.model_name](args, dataset).to(dataset.device)
    dmv = dmvs[args.model_name](dataset.device)
    return {'model': model,
            'dmv': dmv}


def get_logger(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(os.path.join(args.save_dir, 'train.log'), 'w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False
    logger.info(args)
    return logger


def create_save_path(args):

    model_name = args.model.model_name
    args.save_dir = args.save_dir + "/{}".format(model_name) + time.strftime("%Y-%m-%d-%H_%M_%S",
                                                                             time.localtime(time.time()))
    if os.path.exists(args.save_dir):
        print(f'Warning: the folder {args.save_dir} exists.')
    else:
        print('Creating {}'.format(args.save_dir))
        os.makedirs(args.save_dir)
    # save the config file and model file.
    import shutil
    shutil.copyfile(args.conf, args.save_dir + "/conf.ini")
    os.makedirs(args.save_dir + "/parser")
    copy_tree("parser/", args.save_dir + "/parser")


