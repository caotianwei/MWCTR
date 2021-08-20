"""
Models
"""
import os
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import roc_auc_score, log_loss
from torch.nn.utils.rnn import \
    pack_padded_sequence, pad_packed_sequence, PackedSequence
import nni
from tqdm import tqdm
import higher
import utils
import gc
import logging
import numpy as np
import copy
from torch.autograd import grad
from meta import make_functional
# import time

class AuxNet(nn.Module):

    def __init__(self, input_dim, aux_layer_1_dim, aux_layer_2_dim):
        super(AuxNet, self).__init__()
        self.input_dim = input_dim
        self.net = DeepModule(input_dim, aux_layer_1_dim, aux_layer_2_dim)

    def forward(self, states, embs, mask):
        inputs = torch.cat([states, embs], dim=-1).view(-1, self.input_dim)
        mask = mask.reshape(-1, 1)
        pred = torch.sigmoid(self.net(inputs))
        return pred[mask].view(-1, 1)


class AuxLoss(nn.Module):

    def forward(self, click_p, noclick_p):
        click_target = torch.ones(click_p.size()).cuda().float()
        noclick_target = torch.zeros(noclick_p.size()).cuda().float()
        pred = torch.cat([click_p, noclick_p], dim=0)
        target = torch.cat([click_target, noclick_target], dim=0)
        return F.binary_cross_entropy(pred, target)


class DeepModule(nn.Module):

    def __init__(self, input_dim, layer_1_dim, layer_2_dim, drop=0.0):
        super(DeepModule, self).__init__()
        if drop > 0.0:
            self.net = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, layer_1_dim),
                nn.Sigmoid(),
                nn.Linear(layer_1_dim, layer_2_dim),
                nn.Dropout(p=drop),
                nn.Sigmoid(),
                nn.Linear(layer_2_dim, 1))
        else:
            self.net = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, layer_1_dim),
                nn.Sigmoid(),
                nn.Linear(layer_1_dim, layer_2_dim),
                nn.Sigmoid(),
                nn.Linear(layer_2_dim, 1))

    def forward(self, x):
        return self.net(x)


class MaskMean(nn.Module):

    def __init__(self, masked_flag):
        super(MaskMean, self).__init__()
        self.masked_flag = masked_flag

    def forward(self, x, flags):
        mask = (flags != self.masked_flag).float()
        return x.sum(dim=1) / mask.sum(dim=1, keepdim=True)


class WeightedMean(nn.Module):

    def __init__(self, masked_flag):
        super(WeightedMean, self).__init__()
        self.masked_flag = masked_flag

    def forward(self, x, flags, weights):
        mask = (flags != self.masked_flag).float()
        weights = F.softmax(weights, dim=-1)
        x = x * weights
        return x.sum(dim=1) / mask.sum(dim=1, keepdim=True)


class InterestModule(nn.Module):

    def __init__(self, emb_dim, att_layer_1_dim,
                 att_layer_2_dim, masked_flag, pos_dim=0):
        super(InterestModule, self).__init__()
        self.emb_dim = emb_dim
        self.masked_flag = masked_flag
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 4, att_layer_1_dim),
            nn.Sigmoid(),
            nn.Linear(att_layer_1_dim, att_layer_2_dim),
            nn.Sigmoid(),
            nn.Linear(att_layer_2_dim, 1))
        self.pos_dim = pos_dim
        if pos_dim > 0:
            self.query_process = nn.Linear(emb_dim + pos_dim, emb_dim)

    def forward(self, query, keys, flags, is_sum=True, pos_emb=None):
        hist_size = keys.shape[1]
        query = query.repeat(1, hist_size)
        query = query.reshape(-1, hist_size, self.emb_dim)
        if self.pos_dim > 0 and pos_emb is not None:
            query = torch.cat([query, pos_emb], dim=-1)
            query = self.query_process(query)
        feats = torch.cat([
            query, keys, query - keys, query * keys
        ], dim=-1)
        outputs = self.net(feats)
        outputs = outputs.reshape(-1, 1, hist_size)
        # mask
        mask = (flags != self.masked_flag)
        mask = mask.unsqueeze(dim=1)
        padding = torch.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = torch.where(mask, outputs, padding)
        # scale
        outputs = outputs / (self.emb_dim ** 0.5)
        # output
        # weights = torch.sigmoid(outputs)
        weights = F.softmax(outputs, dim=-1)
        if is_sum:
            return (weights@keys).squeeze(dim=1)
        else:
            a, b, c = weights.shape
            return weights.reshape((a, c, b))


class AUGRUCell(nn.RNNCellBase):
    """ 
    Effect of GRU with attentional update gate (AUGRU)    
    """

    def __init__(self, input_dim, hidden_dim, bias=True):
        super(AUGRUCell, self).__init__(
            input_dim, hidden_dim, bias, num_chunks=3)

    def forward(self, inputs, weights, hx=None):
        gi = F.linear(inputs, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)

        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_state = torch.tanh(i_n + reset_gate * h_n)

        # weights = weights.view(-1, 1)
        update_gate = weights * update_gate
        hy = (1.0 - update_gate) * hx + update_gate * new_state
        return hy


class AUGRU(nn.Module):

    def __init__(self, input_dim, hidden_dim, bias=True):
        super(AUGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell = AUGRUCell(input_dim, hidden_dim, bias)

    def forward(self, inputs, weights, hx=None):

        inputs, batch_sizes, sorted_indices, unsorted_indices = inputs
        weights = weights[0]

        max_batch_size = int(batch_sizes[0])
        if hx is None:
            hx = torch.zeros(max_batch_size, self.hidden_dim,
                             dtype=inputs.dtype).cuda()

        outputs = torch.zeros(inputs.size(0), self.hidden_dim,
                              dtype=inputs.dtype).cuda()
        begin = 0
        for batch in batch_sizes:
            new_hx = self.cell(
                inputs[begin:begin + batch],
                weights[begin:begin + batch],
                hx[0:batch])
            outputs[begin:begin + batch] = new_hx
            hx = new_hx
            begin += batch
        return PackedSequence(outputs, batch_sizes,
                              sorted_indices, unsorted_indices)


class SelfMultiHeadAttention(nn.Module):

    def __init__(self, num_units, num_heads,
                 drop_rate, is_layer_norm=True, multi_head_out=False):
        super(SelfMultiHeadAttention, self).__init__()
        self.Q_K_V_layer = nn.Linear(num_units, 3 * num_units)
        self.dropout_1 = nn.Dropout(drop_rate)
        if not multi_head_out:
            self.out_layer = nn.Linear(num_units, num_units)
            self.dropout_2 = nn.Dropout(drop_rate)
            self.layer_norm = nn.LayerNorm(num_units)
        else:
            head_units = num_units // num_heads
            self.out_layers = nn.ModuleList(
                [nn.Linear(head_units, num_units) for _ in range(num_heads)]
            )
            self.dropout_2s = nn.ModuleList(
                [nn.Dropout(drop_rate) for _ in range(num_heads)]
            )
            self.layer_norms = nn.ModuleList(
                [nn.LayerNorm(num_units) for _ in range(num_heads)]
            )
        self.is_layer_norm = is_layer_norm
        self.num_heads = num_heads
        self.num_units = num_units
        self.multi_head_out = multi_head_out

    def forward(self, inputs):
        # attentive operation
        Q_K_V = self.Q_K_V_layer(inputs)
        Q, K, V = Q_K_V.chunk(3, dim=-1)
        Q_ = torch.cat(Q.chunk(self.num_heads, dim=2), dim=0)
        K_ = torch.cat(K.chunk(self.num_heads, dim=2), dim=0)
        V_ = torch.cat(V.chunk(self.num_heads, dim=2), dim=0)
        outputs = Q_@K_.transpose(2, 1)
        align = outputs / (self.num_units ** 0.5)
        # mask
        diag_val = torch.ones_like(align[0, :, :]).cuda()
        tril = torch.tril(diag_val)
        key_masks = tril.expand_as(align)
        padding = torch.ones_like(key_masks).cuda() * (-2 ** 32 + 1)
        outputs = torch.where(key_masks == 0, padding, align)
        # output
        outputs = torch.softmax(outputs, dim=-1)
        outputs = self.dropout_1(outputs)
        outputs = outputs@V_
        split_outputs = outputs.chunk(self.num_heads, dim=0)

        if not self.multi_head_out:
            outputs = torch.cat(split_outputs, dim=2)
            outputs = self.out_layer(outputs)
            outputs = self.dropout_2(outputs)
            outputs = outputs + inputs
            if self.is_layer_norm:
                outputs = self.layer_norm(outputs)
            return outputs
        else:
            results = []
            for i, output in enumerate(split_outputs):
                output = self.out_layers[i](output)
                output = self.dropout_2s[i](output)
                output += inputs
                if self.is_layer_norm:
                    output = self.layer_norms[i](output)
                results.append(output)
            return results


class Model(nn.Module):
    """
    Base class of the pCTR models in our experiments
    """

    def __init__(self, user_count, item_count, cate_count,
                 cate_list, emb_dim):
        super(Model, self).__init__()
        # self.user_emb_w = nn.Embedding(
        #     user_count + 1, emb_dim, user_count)
        self.item_emb_w = nn.Embedding(
            item_count + 1, emb_dim // 2, item_count)
        self.cate_emb_w = nn.Embedding(
            cate_count + 1, emb_dim // 2, cate_count)
        self.item_bias = nn.Parameter(torch.zeros(
            item_count + 1, 1, requires_grad=True))
        self.cate_list = torch.LongTensor(
            cate_list + [cate_count]).cuda()
        self.emb_dim = emb_dim

    def forward(self, x, attention=None):
        _, item, hist_item, neg_hist_item = x
        # Lookup item embedding
        cate = self.cate_list[item]
        item_emb = torch.cat([
            self.item_emb_w(item),
            self.cate_emb_w(cate)
        ], dim=1)
        # Lookup history embedding
        hist_cate = self.cate_list[hist_item]
        hist_emb = torch.cat([
            self.item_emb_w(hist_item),
            self.cate_emb_w(hist_cate)
        ], dim=2)
        # Produce user embedding
        user_emb = self._user_forward(
            hist_item,
            neg_hist_item,
            hist_emb,
            item_emb,
            attention)
        # Output final prediction
        wide_out = self._wide_forward(
            user_emb,
            item_emb)
        deep_out = self._deep_forward(
            user_emb,
            item_emb)
        bias = self.item_bias[item]
        logit = bias + wide_out + deep_out
        return torch.sigmoid(logit)

    def _user_forward(self, hist_item, neg_hist_item, hist_emb, item_emb):
        raise NotImplementedError

    def _wide_forward(self, user_emb, item_emb):
        raise NotImplementedError

    def _deep_forward(self, user_emb, item_emb):
        raise NotImplementedError


class DIN_Base(Model):

    def __init__(self, user_count, item_count, cate_count,
                 cate_list, emb_dim, layer_1_dim, layer_2_dim,
                 aux_layer_1_dim, aux_layer_2_dim, drop=0.0):
        super(DIN_Base, self).__init__(user_count, item_count, cate_count,
                                       cate_list, emb_dim)
        self.deep = DeepModule(emb_dim * 2, layer_1_dim, layer_2_dim, drop=drop)
        # self.attention = InterestModule(
        #     emb_dim, aux_layer_1_dim, aux_layer_2_dim, item_count)

    def _user_forward(self, hist_item, neg_hist_item, hist_emb, item_emb, attention):
        return attention(item_emb, hist_emb, hist_item)

    def _wide_forward(self, user_emb, item_emb):
        return 0

    def _deep_forward(self, user_emb, item_emb):
        deep_input = torch.cat([user_emb, item_emb], dim=1)
        return self.deep(deep_input)


class DIN(nn.Module):
    def __init__(self, user_count, item_count, cate_count,
                 cate_list, emb_dim, layer_1_dim, layer_2_dim,
                 aux_layer_1_dim, aux_layer_2_dim, drop=0.0):
        super(DIN, self).__init__()
        self.base = DIN_Base(user_count, item_count, cate_count,
                             cate_list, emb_dim, layer_1_dim, layer_2_dim,
                             aux_layer_1_dim, aux_layer_2_dim, drop)
        # self.base = DIN_Base(user_count, item_count, cate_count,
        #                      cate_list, emb_dim, layer_1_dim, layer_2_dim,
        #                      aux_layer_1_dim, aux_layer_2_dim)
        self.attention = InterestModule(emb_dim, aux_layer_1_dim,
                                        aux_layer_2_dim, item_count)

    def forward(self, x):
        return self.base(x, self.attention)


class WrappedModel:
    """
    Wrapped model
    """

    def __init__(self, model, optimizer, fine_tune_opt, scheduler,
                 grad_max_norm, critrion=F.binary_cross_entropy):
        self.model = model.cuda()
        self.optimizer = optimizer
        self.fine_tune_opt = fine_tune_opt  # not used
        self.scheduler = scheduler
        self.grad_max_norm = grad_max_norm
        self.critrion = critrion

    def train_step(self, bx, by):
        self.model.train()
        self.optimizer.zero_grad()
        pred = self.model(bx)
        loss = self.critrion(pred, by)
        if hasattr(self.model, 'aux_loss'):
            loss = loss + self.model.aux_loss
        loss.backward()
        clip_grad_norm_(self.model.parameters(),
                        max_norm=self.grad_max_norm)
        self.optimizer.step()
        return loss.item()
        # return loss.item()

    def train(self, train_loader, val_loader,
              test_loader, epoch_num=30):
        n_step = 0

        for i in range(epoch_num):
            # start_time = time.time()
            train_loss = []
            for bx, by in tqdm(train_loader):
                bx, by = [x.cuda() for x in bx], by.cuda()
                loss = self.train_step(bx, by)
                self.scheduler.step()
                n_step += 1
                # if n_step % 1000 == 0:
                #     gc.collect()
                train_loss.append(loss)
            # end_time = time.time()
            train_loss = np.mean(train_loss)
            val_metric = self.test(val_loader)
            test_metric = self.test(test_loader)
            if i > 0 and i % 10 == 0:
                gc.collect()
            # nni.report_intermediate_result(test_metric)
            print('Temp-Batch-loss:{}, Val:{}, Test:{}'.format(
                train_loss, val_metric, test_metric), flush=True)
            # nni.report_intermediate_result(val_metric - test_metric)
        # nni.report_final_result(test_metric)

        # for bx, by in tqdm(val_loader):
        #     bx, by = [x.cuda() for x in bx], by.cuda()
        #     self.model.train()
        #     self.optimizer.zero_grad()
        #     pred = self.model(bx)
        #     loss = F.binary_cross_entropy(pred, by)
        #     loss.backward()
        #     self.optimizer.step()

        test_metric_ft = self.test(test_loader)
        print('Final Test:{}'.format(test_metric_ft), flush=True)
        # nni.report_final_result(val_metric - test_metric)

    @torch.no_grad()
    def test(self, test_loader, metric=roc_auc_score):
        self.model.eval()
        preds = []
        labels = []
        for bx, by in test_loader:
            bx, by = [x.cuda() for x in bx], by.cuda()
            pred = self.model(bx)
            preds.append(pred)
            labels.append(by)
        preds = torch.cat(preds, dim=0).cpu().numpy()
        labels = torch.cat(labels, dim=0).cpu().numpy()
        return metric(labels, preds)
        # preds = torch.cat(preds, dim=0)
        # labels = torch.cat(labels, dim=0)
        # return F.binary_cross_entropy(preds, labels).item()

    def save(self, model_dir):
        fname = '{}_{}_{}_{}.pth'.format(
            self.__class__.__name__,
            self.model.__class__.__name__,
            nni.get_experiment_id(),
            nni.get_trial_id()
        )
        model_path = os.path.join(model_dir, fname)
        print('model_path', model_path, flush=True)
        torch.save(self.model.state_dict(), model_path)

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))


class MetaLearingModel(WrappedModel):

    def __init__(self, inner_lr, inner_steps, mu,
                 *args, **kargs):
        super(MetaLearingModel, self).__init__(*args, **kargs)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.mu = mu

    def train(self, train_loader, val_loader,
              test_loader, epoch_num=50):
        # meta-data
        self.inf_val_loader = utils.inf_loader(val_loader)
        super(MetaLearingModel, self).train(
            train_loader, val_loader, test_loader, epoch_num)

    def get_base_params(self, model):
        params = set(model.parameters())
        att_params = set(model.attention.parameters())
        return list(params - att_params)

    def _detach_base(self, fmodel):
        base_params = self.get_base_params(fmodel)

        for p in base_params:
            p.detach_()
        for p in fmodel.deep.parameters():
            p.requires_grad_()
        fmodel.item_bias.requires_grad_()

    def train_step(self, bx, by):
        self.model.train()
        # inner_model = self.model.base
        # inner_opt = SGD(inner_model.parameters(), self.inner_lr)
        self.optimizer.zero_grad()
        # vx, vy = next(self.inf_val_loader)
        # b_size = int(len(bx[0]) * 0.5)
        b_size = int(len(bx[0]) * 0.8)
        vx = [x[b_size:] for x in bx]
        vy = by[b_size:]
        bx = [x[:b_size] for x in bx]
        by = by[:b_size]

        pred_b = self.model(bx)
        loss_b = self.critrion(pred_b, by)
        b_grad = grad(loss_b, self.model.base.parameters(), create_graph=True)

        fast_weights = list(map(lambda p: p[1] - self.inner_lr * p[0],
                                zip(b_grad, self.model.base.parameters())))
        f_model = make_functional(self.model.base)

        for _ in range(self.inner_steps - 1):
           fast_weights = list(map(lambda p: p[1] - self.inner_lr * p[0],
                                zip(b_grad, fast_weights)))

        pred_v = f_model(vx, self.model.attention, params=fast_weights)
        loss_v = self.critrion(pred_v, vy)

        loss = loss_b + self.mu * loss_v
        loss.backward()
        clip_grad_norm_(self.model.parameters(),
                        max_norm=self.grad_max_norm)
        self.optimizer.step()
        return loss.item()

