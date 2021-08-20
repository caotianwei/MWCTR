"""
factory functions
"""
from torch import optim
from torch.optim import lr_scheduler
import models

def _produce(conf, module, name, params):
    cls_name = getattr(module, conf[name])
    return cls_name(**params)


def get_model(conf, data):
    params = {
        'user_count': data.user_count,
        'item_count': data.item_count,
        'cate_count': data.cate_count,
        'cate_list': data.cate_list,
        'emb_dim': conf['emb_dim'],
        'layer_1_dim': conf['layer_1_dim'],
        'layer_2_dim': conf['layer_2_dim']
    }
    if conf['model'] == 'DIN':
        params['aux_layer_1_dim'] = conf['aux_layer_1_dim']
        params['aux_layer_2_dim'] = conf['aux_layer_2_dim']
        if 'drop' in conf.keys():
            params['drop'] = conf['drop']
    if conf['model'] == 'DIEN':
        params['max_len'] = data.max_len
        params['aux_layer_1_dim'] = conf['aux_layer_1_dim']
        params['aux_layer_2_dim'] = conf['aux_layer_2_dim']
    if conf['model'] == 'DMIN':
        params['max_len'] = data.max_len
        params['aux_layer_1_dim'] = conf['aux_layer_1_dim']
        params['aux_layer_2_dim'] = conf['aux_layer_2_dim']
        params['pos_emb_dim'] = conf['pos_emb_dim']
        params['num_head'] = conf['num_head']
        params['mh_drop_rate'] = conf['mh_drop_rate']

    return _produce(conf, models, 'model', params)


def get_optimizer(conf, model_params, fine_tune=False):
    params = {
        'params': model_params,
        'lr': conf['lr'],
        'weight_decay': conf['weight_decay']
    }
    if fine_tune:
        params['lr'] = conf['fine_tune_lr']
    if 'nesterov' in conf.keys():
         params['nesterov'] = eval(conf['nesterov'])
    if 'momentum' in conf.keys():
         params['momentum'] = conf['momentum']
    return _produce(conf, optim, 'optim', params)


def get_scheduler(conf, opt):
    if isinstance(conf['milestones'], int):
        milestones = [conf['milestones']]
    else:
        milestones = eval(conf['milestones'])
    params = {
        'optimizer': opt,
        'milestones': milestones,
        'gamma': conf['gamma'],
    }
    return _produce(conf, lr_scheduler, 'lr_scheduler', params)


def get_wrapt_model(conf, data):
    model = get_model(conf, data)
    optimizer = get_optimizer(conf, model.parameters(), False)
    fine_tune_opt = get_optimizer(conf, model.parameters(), True)
    scheduler = get_scheduler(conf, optimizer)
    # produce wrapt model
    if conf['wrap'] == 'WrappedModel':
        return models.WrappedModel(
            model, 
            optimizer, 
            fine_tune_opt, 
            scheduler, 
            conf['grad_max_norm'])
    if conf['wrap'] == 'MetaLearingModel':
        return models.MetaLearingModel(
            conf['inner_lr'], 
            conf['inner_steps'], 
            conf['mu'],
            model, 
            optimizer, 
            fine_tune_opt,
            scheduler, 
            conf['grad_max_norm'])
    if conf['wrap'] == 'GDAModel':
        return models.GDAModel(
            conf['inner_lr'], 
            conf['inner_steps'], 
            conf['mu'],
            model, 
            optimizer, 
            fine_tune_opt,
            scheduler, 
            conf['grad_max_norm'])
    if conf['wrap'] == 'ValModel':
        return models.ValModel(
            conf['inner_lr'], 
            conf['inner_steps'], 
            conf['mu'],
            model, 
            optimizer, 
            fine_tune_opt,
            scheduler, 
            conf['grad_max_norm'])
    raise RuntimeError('bad config')
