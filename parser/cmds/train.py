# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
from parser.cmds.cmd import CMD
from parser.helper.metric import Metric
from parser.helper.util import *
from parser.helper.loader_wrapper import LoaderWrapper
from parser.helper.data_module import DataModule
from parser.trainer import KM_initializer, External_parser_initializer
import torch
from parser.const import *

class Train(CMD):


    def __call__(self, args):
        self.args = args
        self.device = args.device
        create_save_path(args)
        dataset = DataModule(args)
        models = get_model(args.model, dataset)
        self.model = models['model']
        self.dmv = models['dmv']
        if args.joint_training:
            self.dmv1o = DMV1o(device=dataset.device)
        log = get_logger(args)
        log.info("Create the model")
        log.info(f"{self.model}\n")
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                              lr=args.optimizer.lr,
                                          betas=(args.optimizer.mu, args.optimizer.nu), weight_decay=args.optimizer.weight_decay)
        total_time = timedelta()
        best_e, best_metric = 1, Metric()
        log.info(self.optimizer)

        '''
        Initialization:
        k&m initialization 
        or 
        use external parser's result to initialize (the result is save as conll file.)
        '''
        if args.train.initializer == 'km':
            initializer = KM_initializer(dataset=dataset)
            initializer.initialize(model=self.model, dmv=self.dmv, loader=dataset.train_init_dataloader, optimizer=self.optimizer, hparams=args.train.init)

        elif args.train.initializer == 'external':
            initializer = External_parser_initializer(device=args.device)
            if not self.args.joint_training:
                train_loader_init = dataset.train_init_dataloader
                initializer.initialize(self.model, train_loader_init, self.optimizer, hparams=args.train.init)
            else:
                initializer.initialize(self.model.model1, dataset.train_init_dataloader, self.optimizer, hparams=args.train.init.model1)
                initializer.initialize(self.model.model2, dataset.train_init_dataloder_for_model2, self.optimizer, hparams=args.train.init.model2)
        else:
            log.info("random initialization~")


        train_loader = dataset.train_dataloader
        eval_loader = dataset.val_dataloader
        test_loader = dataset.test_dataloader

        self.model.eval()
        test_loader_autodevice = LoaderWrapper(test_loader, device=self.device)
        test_metric = self.evaluate(test_loader_autodevice)
        log.info(f"initialize: {'test:':6}  {test_metric}")

        '''
        Training
        '''
        train_arg = args.train.training


        for epoch in range(1, train_arg.max_epoch + 1):
            '''
            Auto .to(self.device)
            '''
            train_loader_autodevice = LoaderWrapper(train_loader, device=self.device)
            eval_loader_autodevice = LoaderWrapper(eval_loader, device=self.device)
            test_loader_autodevice = LoaderWrapper(test_loader, device=self.device)
            start = datetime.now()
            train_metric = self.train(train_loader_autodevice)
            log.info(f"Epoch {epoch} / {train_arg.max_epoch}:")
            log.info(f"{'train:':6}   {train_metric}")
            dev_metric = self.evaluate_likelihood(eval_loader_autodevice)
            log.info(f"{'dev:':6}   {dev_metric}")
            test_metric = self.evaluate(test_loader_autodevice)
            log.info(f"{'test:':6}  {test_metric}")

            if args.joint_training:
                test_loader_autodevice = LoaderWrapper(test_loader, device=self.device)
                test_metric_second_order = self.evaluate(test_loader_autodevice, dmv=self.dmv, model=self.model.model2)
                log.info(f"{'second-order: test:':6}  {test_metric_second_order}")
                test_loader_autodevice = LoaderWrapper(test_loader, device=self.device)
                test_metric_first_order = self.evaluate(test_loader_autodevice, dmv=self.dmv1o, model=self.model.model1)
                log.info(f"{'first-order: test:':6}  {test_metric_first_order}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                torch.save(
                   obj=self.model.state_dict(),
                   f = args.save_dir + "/best.pt"
                )
                log.info(f"{t}s elapsed (saved)\n")
            else:
                log.info(f"{t}s elapsed\n")

            torch.save(
                {'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                },  args.save_dir + "/latest.pt"
            )
            total_time += t
            if epoch - best_e >= train_arg.patience:
                break

