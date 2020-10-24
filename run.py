# -*- coding: utf-8 -*-

import argparse
import os
from parser.cmds import Evaluate, Predict, Train
import shutil
import torch
import traceback
from pathlib import Path
from easydict import EasyDict as edict
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create the Biaffine Parser model.'
    )
    parser.add_argument('--conf', '-c', default='data/config.ini')
    parser.add_argument('--device', '-d', default='1')
    parser.add_argument('--mode', '-m', default='train')

    args2 = parser.parse_args()
    yaml_cfg = yaml.load(open(args2.conf, 'r'))
    args = edict(yaml_cfg)
    args.update(args2.__dict__)

    subcommands = {
        'evaluate': Evaluate(),
        'predict': Predict(),
        'train': Train()
    }
    print(f"Set the device with ID {args.device} visible")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Run the subcommand in mode {args.mode}")
    cmd = subcommands[args.mode]
    config_path = Path(args.conf)
    config_name = config_path.stem

    try:
        cmd(args)
    except KeyboardInterrupt:
        command = int(input('Enter 0 to delete the repo, and enter anything else to save.'))
        if command == 0:
            shutil.rmtree(args.save_dir)
            print("You have successfully delete the created log directory.")
        else:
            print("log directory have been saved.")
    except Exception:
        traceback.print_exc()
        print("log directory have been saved.")


