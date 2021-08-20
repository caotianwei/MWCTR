"""
Main()
"""
import trails
import sys

if __name__ == "__main__":

    data_name = sys.argv[1]
    
    if data_name == 'Electronics':
        conf = {'seed': 1234, 'data_path': 'data_1_1/amazon_dataset_1_1.pkl', 'wrap': 'MetaLearingModel', 'model': 'DIN', 'emb_dim': 128, 'layer_1_dim': 80, 'layer_2_dim': 40, 'optim': 'SGD', 'aux_layer_1_dim': 80, 'aux_layer_2_dim': 40, 'lr': 1, 'weight_decay': 0, 'lr_scheduler': 'MultiStepLR', 'pos_emb_dim': 2, 'num_head': 2, 'mh_drop_rate': 0, 'milestones': 42000, 'gamma': 0.1, 'grad_max_norm': 5, 'fine_tune_lr': 1e-05, 'nesterov': 'True', 'momentum': 0.9, 'train_batch_size': 256, 'test_batch_size': 512, 'epoch_num': 30, 'model_dir': 'saved_model', 'inner_lr': 0.01, 'inner_steps': 1, 'mu': 0.4}  
    elif data_name == 'Books':
        conf = {'seed': 1234, 'data_path': 'data_1_1/books_dataset_1_1.pkl', 'wrap': 'MetaLearingModel', 'model': 'DIN', 'emb_dim': 128, 'layer_1_dim': 80, 'layer_2_dim': 40, 'optim': 'SGD', 'lr': 1, 'aux_layer_1_dim': 80, 'aux_layer_2_dim': 40, 'weight_decay': 0, 'lr_scheduler': 'MultiStepLR', 'milestones': 42000, 'gamma': 0.1, 'grad_max_norm': 5, 'fine_tune_lr': 1e-05, 'train_batch_size': 40, 'test_batch_size': 512, 'epoch_num': 30, 'model_dir': 'saved_model', 'inner_lr': 0.01, 'inner_steps': 1, 'mu': 0.4}
    elif data_name == 'Games':
        conf = {'seed': 1234, 'data_path': 'data_1_1/games_dataset_1_1.pkl', 'wrap': 'MetaLearingModel', 'model': 'DIN', 'emb_dim': 16, 'layer_1_dim': 80, 'layer_2_dim': 40, 'optim': 'SGD', 'aux_layer_1_dim': 60, 'aux_layer_2_dim': 30, 'lr': 0.5, 'momentum': 0.8, 'nesterov': 'True', 'weight_decay': 1e-05, 'lr_scheduler': 'MultiStepLR', 'pos_emb_dim': 2, 'num_head': 2, 'mh_drop_rate': 0, 'milestones': 27948, 'gamma': 0.1, 'grad_max_norm': 5, 'fine_tune_lr': 1e-05, 'train_batch_size': 128, 'test_batch_size': 512, 'epoch_num': 30, 'model_dir': 'saved_model', 'inner_lr': 0.01, 'inner_steps': 1, 'mu': 0.1}
    elif data_name == 'Taobao':
        conf = {'seed': 1234, 'data_path': 'data_1_1/taobao_dataset_1_1.pkl', 'wrap': 'MetaLearingModel', 'model': 'DIN', 'emb_dim': 128, 'layer_1_dim': 80, 'layer_2_dim': 40, 'optim': 'SGD', 'aux_layer_1_dim': 80, 'aux_layer_2_dim': 40, 'lr': 1, 'weight_decay': 0, 'lr_scheduler': 'MultiStepLR', 'pos_emb_dim': 2, 'num_head': 2, 'mh_drop_rate': 0, 'milestones': 42000, 'gamma': 0.1, 'grad_max_norm': 5, 'fine_tune_lr': 1e-05, 'train_batch_size': 256, 'test_batch_size': 512, 'epoch_num': 30, 'model_dir': 'saved_model', 'inner_lr': 0.01, 'inner_steps': 1, 'mu': 0.4}
    elif data_name == 'MovieLens':
        conf = {'seed': 1234, 'data_path': 'data_1_1/mvlens_dataset_1_1.pkl', 'wrap': 'MetaLearingModel', 'model': 'DIN', 'emb_dim': 64, 'layer_1_dim': 80, 'layer_2_dim': 40, 'aux_layer_1_dim': 80, 'aux_layer_2_dim': 40, 'optim': 'SGD', 'lr': 1, 'weight_decay': 0.0001, 'lr_scheduler': 'MultiStepLR', 'milestones': '[17087, 61025]', 'gamma': 0.1, 'nesterov': 'True', 'momentum': 0.9, 'grad_max_norm': 5, 'fine_tune_lr': 1e-05, 'train_batch_size': 512, 'test_batch_size': 512, 'epoch_num': 30, 'model_dir': 'saved_model', 'inner_lr': 0.001, 'inner_steps': 1, 'mu': 0.001}
    else:
        raise RuntimeError('Dataset Not Found!')
    
    trail = trails.Trail(conf)
    trail.run_train()
