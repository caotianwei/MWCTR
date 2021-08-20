"""
Trail
"""
import torch
import nni
import datasets
import utils
import factory


class Trail:

    def __init__(self, conf):
        torch.cuda.set_device(6)
        print(conf, flush=True)
        utils.setup_seed(conf['seed'])
        self.conf = conf
        
        if 'hp' in conf.keys():
            self.data = datasets.Data(conf['data_path'], hp=conf['hp'])
        else:
            self.data = datasets.Data(conf['data_path'])

        self.wrapped_model = factory.get_wrapt_model(conf, self.data)

    def run_train(self):
        train_loader = self.data.produce_loader(
            'train_set',
            self.conf['train_batch_size'],
        )
        val_loader = self.data.produce_loader(
            'val_set',
            # 'train_set',
            self.conf['test_batch_size'],
        )
        test_loader = self.data.produce_loader(
            'test_set',
            self.conf['test_batch_size'],
            False
        )
        self.wrapped_model.train(
            train_loader,
            val_loader,
            test_loader,
            self.conf['epoch_num']
        )
        self.wrapped_model.save(self.conf['model_dir'])

    def run_test(self, model_path):
        self.wrapped_model.load(model_path)
        test_loader = self.data.produce_loader(
            'test_set',
            self.conf['test_batch_size'],
            False
        )
        metirc_value = self.wrapped_model.test(test_loader)
        nni.report_final_result(metirc_value)
