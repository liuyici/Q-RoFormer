"""
Q-RoFormer: Quaternion Rotation-based
Transformer for cross-subject EEG emotion
recognition
"""


import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 使用第2号物理GPU
import numpy as np
import math
import random
import datetime
import time
import scipy.io
from modules import PseudoLabeledData, load_seed, load_seed_raw, fine_tuning_load_XY,load_seed_three_feature
from dataloader import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import Parameter
# import lr_schedule
from   torch                            import autograd
from   torch.autograd                   import Variable
from   core_qnn.quaternion_layers       import *
import torchvision.transforms as transforms
import utils
from utils import LabelSmooth
import Adver_network
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import pynvml
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
# from torch.backends import cudnn
# cudnn.benchmark = False
# cudnn.deterministic = True
class SLR_layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SLR_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        r = input.norm(dim=1).detach()[0]
        cosine = F.linear(input, F.normalize(self.weight), r * torch.tanh(self.bias))
        output = cosine
        return output

def GaussianNoise(x, sigma = 1.0):
    noise = torch.tensor(0.0).cuda()
    sampled_noise = noise.repeat(*x.size()).normal_(mean=0, std=sigma)
    x = x + sampled_noise
    return x

class DFN_LSTM(nn.Module):
    def __init__(self, emb_size=40):
    # def __init__(self, input_size = 2790, hidden_size = 320, use_bottleneck=True, bottleneck_dim=256, radius=10.0, class_num=3):
        super(DFN_LSTM, self).__init__()
        self.qlstm = QLSTM(310, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(310, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.bottleneck = nn.Sequential(
            nn.Linear(128 * 2, 128),  # 双向LSTM输出 size 是 2*hidden_size
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.radius = 10
        self.fc = SLR_layer(128, 3)  # 分类层
        self.__in_features = 128
    def forward(self, x):
        # if self.training:
        #     x = GaussianNoise(x, sigma=1.0)
        # batch_size,feature_step, time_steps = x.shape  # [100, 310, 3]
        # x = x.view(batch_size, time_steps, feature_step)  # 合并通道和频带 -> [100, 3, 310]
        #lstm_out, _ = self.qlstm(x)  # LSTM 输出 [batch, time_steps, hidden_size*2]
        lstm_out, _ = self.lstm(x)  # LSTM 输出 [batch, time_steps, hidden_size*2]
        # print(lstm_out.shape)  # [batch, time_steps, hidden_size*2]  [32, 3, 256]
        x = lstm_out[:, -1, :]  # 取最后一个时间步的输出  [batch, hidden_size*2]

        x = self.bottleneck(x)  # 经过 bottleneck
        x = self.radius * x / (torch.norm(x, dim=1, keepdim=True) + 1e-10)  # 归一化
        y = self.fc(x)  # 分类

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = QuaternionLinearAutograd(308, 308, bias=False,
                 init_criterion='glorot', weight_init='quaternion',
                 seed=None, rotation=True, quaternion_format=True, scale=False)
        self.queries = nn.Linear(308, 308)
        self.values = nn.Linear(308, 308)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        res_fea = x[:, :, -2:]    # [32, 3, 2]
        # print(res_fea.shape)
        q_x = x[:, :, :-2]          # [32, 3, 308]
        # print(q_x.shape)
        # q_x = self.layer1(q_x)
        # print(q_x.shape)
        x = torch.cat([q_x, res_fea], dim=-1)  # [32, 12, 310]
        queries = rearrange(self.queries(q_x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(q_x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(q_x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        
        out = rearrange(out, "b h n d -> b n (h d)")
        new_x = torch.cat([out, res_fea], dim=-1)
        out = self.projection(new_x)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=14,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, bottleneck_dim, n_classes):
        super().__init__()
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            SLR_layer(32, n_classes)
        )
        
    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc2(x)
        
        return x, out

class Discriminator(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.fc2 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            SLR_layer(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc2(x)
        
        return out

# 
class RoFormer(nn.Sequential):
    def __init__(self, emb_size=10, depth=6, bottleneck_dim=256, n_classes=4, **kwargs):
        super().__init__(   
            #**(depth, emb_size),# 输入到这里是【批次，310，3】
            #**(emb_size),# 改为：输入到这里是【批次，3，310】
            ClassificationHead(emb_size, bottleneck_dim, n_classes)
        )


class ExGAN():
    def __init__(self, args, nsub, fold):
        super(ExGAN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.batch_size = 50
        self.n_epochs = 60  #1000
        self.img_height = 22
        self.img_width = 600
        self.channels = 1
        self.c_dim = 4
        self.lr = 0.002
        self.lr2 = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.alpha = 0.0002
        self.dimension = (190, 50)
        self.nSub = nsub
        self.radius = 10
        self.start_epoch = 0
        self.root = '/home/lyc/research/research_5/DA/ExtractedFeatures/'

        self.pretrain = False

        self.log_write = open("/home/lyc/research/research_6/EEG-Transformer-424/snapshot.txt", "w")

        self.img_shape = (self.channels, self.img_height, self.img_width)

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = RoFormer(emb_size=310, depth=6, bottleneck_dim=128, n_classes=3).cuda()
        self.domain_Discriminator = Discriminator(emb_size=10, n_classes=14).cuda()
        self.criterion = LabelSmooth(num_class=args.num_class).cuda()

    def get_source_data(self, feature="de_LDS"):
        if self.args.dataset == "seed":
            train_dataset, test_dataset, X, Y = load_seed(self.args, self.args.file_path, session=self.args.session, feature=feature)
        return train_dataset, test_dataset, X, Y 
    
    def get_source_data_for_fine(self, X, Y):
        if self.args.dataset == "seed":
            dset_loaders = fine_tuning_load_XY(self.args, X, Y)
        return dset_loaders

    def test_suda(self, loader, model):
        start_test = True
        with torch.no_grad():
            # 获得迭代数据
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                # 获得样本与标签
                data = next(iter_test)
                inputs = data[0]
                labels = data[1]
                # 使用gpu
                inputs = inputs.type(torch.FloatTensor).cuda()
                inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # 自动计算 62×5=310 [批次，3，310]
                # inputs = inputs.permute(0, 2, 1) 
                labels = labels
                # 获得预测结果
                _, outputs = model(inputs)
                # 200个批次连接
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        # 获得预测标签
        _, predictions = torch.max(all_output, 1)
        # 计算所有样本的acc
        accuracy = torch.sum(torch.squeeze(predictions).float() == all_label).item() / float(all_label.size()[0])
    
        y_true = all_label.cpu().data.numpy()
        y_pred = predictions.cpu().data.numpy()
        labels = np.unique(y_true)
    
        # 计算各种指标
        ytest = label_binarize(y_true, classes=labels)
        ypreds = label_binarize(y_pred, classes=labels)
    
        f1 = f1_score(y_true, y_pred, average='macro')
        auc = roc_auc_score(ytest, ypreds, average='macro', multi_class='ovr')
        matrix = confusion_matrix(y_true, y_pred)
    
        return accuracy, f1, auc, matrix

    def train(self, fold):

        train_dataset, test_dataset, X, Y  = self.get_source_data(feature="de_LDS")
        self.optimizer = torch.optim.SGD(
                         list(self.model.parameters()) + list(self.domain_Discriminator.parameters()),  # 同时优化model和domain_Discriminator的参数
                         lr=self.lr,
                         momentum=0.9,
                         weight_decay=0.005
                        )
        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        for e in range(self.n_epochs):
            self.model.train()
            for i, data in enumerate(train_dataset):
                x_src = list()
                y_src = list()
                d_src = list()
                index = 0
                for domain_idx in range(15 - 1):
                    tmp_x = data['Sx' + str(domain_idx + 1)].float().cuda()
                    tmp_y = data['Sy' + str(domain_idx + 1)].long().cuda()
                    labels = torch.from_numpy(np.array([[index] * args.batch_size]).T).type(torch.FloatTensor).flatten().long().cuda()
                    x_src.append(tmp_x)
                    d_src.append(labels)
                    y_src.append(tmp_y)
                    index += 1
                x_trg = data['Tx'].float().cuda()
                test_label = data['Ty'].long().cuda()
                
                img = torch.cat(x_src, dim=0)
               
                label = torch.cat(y_src, dim=0)
                domain_label = torch.cat(d_src, dim=0)
                img = img.view(img.size(0), img.size(1),  -1)  # 自动计算 62×5=310 [批次，3，310]
                x_trg = x_trg.view(x_trg.size(0), x_trg.size(1), -1)  # 自动计算 62×5=310 [批次，3，1，310]
                tok, outputs = self.model(img)
                tok_target, outputs_target = self.model(x_trg)
                pre_target = torch.nn.functional.softmax(outputs_target,dim = 1)
                
                mmd_b_loss = 0 
                mmd_t_loss = 0
                pred_src_domain_D = []
                target_list = []
                target_pre_list = []
                for i in range(14):
                    target_list.append(tok_target)
                    target_pre_list.append(pre_target)
                # 拼接
                tok_target = torch.cat(target_list, dim=0)  # [504, 280]
                output_target = torch.cat(target_pre_list, dim=0)
                
                mmd_b_loss = utils.marginal(tok,tok_target)
                mmd_t_loss += utils.conditional(
                    tok,
                    tok_target,
                    label.reshape((args.batch_size*14, 1)),
                    output_target,
                    2.0,
                    5,
                    None)
                features_s_Adver = Adver_network.ReverseLayerF.apply(tok, args.gamma)#用这个替代features_source经过了反转层
                outputs_D = self.domain_Discriminator(features_s_Adver)
                Adver_domain_labels_loss = self.criterion_cls(outputs_D, domain_label.flatten())

                if isinstance(mmd_b_loss, np.ndarray):
                     mmd_b_loss = torch.tensor(mmd_b_loss, device=outputs.device, dtype=torch.float32)

                if isinstance(mmd_t_loss, np.ndarray):
                    mmd_t_loss = torch.tensor(mmd_t_loss, device=outputs.device, dtype=torch.float32)
                if isinstance(Adver_domain_labels_loss, np.ndarray):
                    Adver_domain_labels_loss = torch.tensor(Adver_domain_labels_loss, device=outputs.device, dtype=torch.float32)
                MMD_loss = 0.5*mmd_b_loss + 0.5*mmd_t_loss

                loss = self.criterion(outputs, label)  + MMD_loss + Adver_domain_labels_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            out_epoch = time.time()

            if (e + 1) % 1 == 0:
                start_test = True
                with torch.no_grad():        
                    self.model.eval()
        
                    for batch_idx, tar_data in enumerate(test_dataset):
                        Tx = tar_data['Tx']
                        Ty = tar_data['Ty']
                        Tx = Tx.float().cuda()
                        Tx = Tx.view(Tx.size(0), Tx.size(1), -1)  # 自动计算 62×5=310 [批次，3，310]
                        # Tx = Tx.permute(0, 2, 1) 
                        Tok, Cls = self.model(Tx)
                        if start_test:
                            all_output = Cls.float().cpu()
                            all_label = Ty.float()
                            start_test = False
                        else:
                            all_output = torch.cat((all_output, Cls.float().cpu()), 0)
                            all_label = torch.cat((all_label, Ty.float()), 0)
                        loss_test = self.criterion_cls(Cls.float().cpu(), Ty.long())
                torch.cuda.empty_cache()  # 清理GPU缓存
                y_pred = torch.max(all_output, 1)[1]
                acc = float((y_pred == all_label).cpu().numpy().astype(int).sum()) / float(all_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
                # print('The epoch is:', e, '  The accuracy is:', acc)
                print('Epoch:', e,
                      '  Train loss: %.4f' % loss.item(),
                      '  Test loss: %.4f' % loss_test.detach().cpu().numpy(),
                      '  Train acc: %.4f' % train_acc,
                      '  Test acc: %.4f' % acc)
                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = Ty
                    Y_pred = y_pred

        averAcc = averAcc / num
        print('The average accuracy of n_epochs%d is:' %(e+1), averAcc)
        print('The best accuracy of n_epochs%d is:' %(e+1), bestAcc)
        self.log_write.write('The average accuracy of n_epochs%d is: ' %(e+1) + str(averAcc) + "\n")
        self.log_write.write('The best accuracy n_epochs%d is: ' %(e+1) + str(bestAcc) + "\n")
        return bestAcc, averAcc, Y_true, Y_pred, X, Y, self.model


    def fine_tuning(self, args, X, Y, model):
        dset_loaders = self.get_source_data_for_fine(X, Y)
        # 获取model参数，这里保持和conformer一致
        parameter_model = model.parameters()
        self.optimizer = torch.optim.Adam(parameter_model, lr=self.lr2, betas=(self.b1, self.b2))
        # self.optimizer = torch.optim.SGD(
        #          parameter_model,  # 同时优化model和domain_Discriminator的参数
        #          lr=self.lr2,
        #          momentum=0.9,
        #          weight_decay=0.0005
        #         )
        # 多显卡训练
        # gpus = args.gpu_id.split(',')
        # if len(gpus) > 1:
        #     model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])
    
        ## 开始训练
        # 获取数据个数
        len_train_source = len(dset_loaders["source"])
        len_train_target = len(dset_loaders["target"])
    
        # a定义变量
        best_acc = 0.0
        final_acc = 0
        final_f1 = 0
        final_auc = 0
        final_mat = []
    
        # 开始训练，先测一下结果，看看是不是正确继承了model
        for i in range(args.max_iter2):
            if i % 1 == 0:
                with torch.no_grad():      
                    model.eval()
                    best_acc, best_f1, best_auc, best_mat = self.test_suda(dset_loaders, model)
                    if final_acc < best_acc:
                        final_acc = best_acc
                        final_f1 = best_f1
                        final_auc = best_auc
                        final_mat = best_mat  
                    if i == 0:
                        log_str = "iter: {:05d}, \t accuracy: {:.4f} \t f1: {:.4f} \t auc: {:.4f}".format(i, best_acc, best_f1, best_auc)
                    else: 
                        log_str = "iter: {:05d}, \t accuracy: {:.4f} \t f1: {:.4f} \t auc: {:.4f} \t loss: {:.4f}".format(i, best_acc, best_f1, best_auc, total_loss.item())
    
                    self.log_write.write(str(i) + "    " + str(best_acc) + "\n")
                    print(log_str)
            model.train()
    
            # self.optimizer = lr_schedule.inv_lr_scheduler(self.optimizer, i, lr=args.lr_b)
          
            # 开始获取批次数据
            if i % len_train_source == 0:
                iter_source = iter(dset_loaders["source"])
            if i % len_train_target == 0:
                iter_target = iter(dset_loaders["target"])
           
            inputs_source_, labels_source = next(iter_source)
            inputs_target_, ture_labels_target = next(iter_target)
            
            inputs_source_ = inputs_source_.type(torch.FloatTensor)
            labels_source = labels_source.type(torch.LongTensor)
            inputs_target_ = inputs_target_.type(torch.FloatTensor)
            ture_labels_target = ture_labels_target.type(torch.LongTensor)
            # 挂到GPU上
            inputs_source, labels_source = inputs_source_.cuda(), labels_source.cuda()
            inputs_target, ture_labels_target = inputs_target_.cuda(), ture_labels_target.cuda()
            inputs_source = inputs_source.view(inputs_source.size(0), inputs_source.size(1), -1)  # 自动计算 62×5=310 [批次，3，310]
            # inputs_source = inputs_source.permute(0, 2, 1) 
            inputs_target = inputs_target.view(inputs_target.size(0), inputs_target.size(1), -1)  # 自动计算 62×5=310 [批次，3，310]
            # inputs_target = inputs_target.permute(0, 2, 1) 
            features_source, outputs_source = model(inputs_source)
            features_target, outputs_target = model(inputs_target)
            # 损失
            classifier_loss = self.criterion_cls(outputs_source, labels_source.flatten())
            pre_target = F.softmax(outputs_target, dim=1)
            ce_loss = torch.mean(utils.Entropy(pre_target))
            # print(labels_source.shape)
            # print(outputs_target.shape)
            # predictions = torch.argmax(pseu_labels_target, dim=1)
            mmd_b_loss = utils.marginal(features_source,features_target)
            mmd_t_loss = utils.conditional(
                       features_source,
                       features_target,
                       labels_source.reshape((200, 1)),
                       torch.nn.functional.softmax(outputs_target,dim = 1),
                       2.0,
                       5,
                       None)
            mmd_loss = 0.5*mmd_b_loss + 0.5*mmd_t_loss
           # CORAL = utils.CORAL_loss(outputs_source, outputs_target)
            total_loss = classifier_loss + 0.1 * ce_loss + mmd_loss
    
            # 传递梯度
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return final_acc, final_f1, final_auc, final_mat, model

def main(args):
    pre_train = []
    tuning = []
    result_write = open("/home/lyc/research/research_6/EEG-Transformer-424\snapshot.txt", "w")

    for i in range(15):
        args.target = 15 - i
        # starttime = datetime.datetime.now()
        seed_n = 1

        result_write.write('--------------------------------------------------')
        # print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        print('Subject %d' % (i+1))
        
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
        ba = 0
        aa = 0
        pre_train_Acc = 0
        averAcc = 0

        exgan = ExGAN(args, i + 1, 1)
        
        ba, aa, _, _, X, Y, model  = exgan.train(1)
        final_acc, final_f1, final_auc, final_mat, model = exgan.fine_tuning(args, X, Y, model)

        result_write.write('pre_training acc is:' + str(ba) + "\n")
        result_write.write('fine_tuning acc is:' + str(final_acc) + "\n")
        pre_train_Acc = ba
        tuning_Acc = final_acc
        # plot_confusion_matrix(Y_true, Y_pred, i+1)
        pre_train.append(pre_train_Acc)
        tuning.append(tuning_Acc)
        # endtime = datetime.datetime.now()
        # print('subject %d duration: '%(i+1) + str(endtime - starttime))
        print('pre_training acc is:', pre_train)
        print('fine_tuning acc is:', tuning)


    pre_ave = sum(pre_train) / len(pre_train)
    tuning_ave = sum(tuning) / len(tuning)
    print('------------------------pre-training result--------------------------', pre_train)
    print('------------------------fin-tuning result--------------------------', tuning)
    print('------------------------pre-training average result--------------------------', pre_ave)
    print('------------------------fin-tuning average result--------------------------', tuning_ave)
    result_write.write('--------------------------------------------------')
    result_write.write(f"All accuracy is: {pre_train}\n")
    result_write.write(f"All subject Aver accuracy is: {tuning}\n")
    result_write.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spherical Space Domain Adaptation with Pseudo-label Loss')
    parser.add_argument('--baseline', type=str, default='MSTN', choices=['MSTN', 'DANN'])
    parser.add_argument('--device', type=str, default='cuda')
    # parser.add_argument('--gpu_id', type=str, nargs='?', default='3', help="device id to run")
    parser.add_argument('--dataset',type=str,default='seed')
    parser.add_argument('--source', type=str, default='amazon')
    parser.add_argument('--target', type=int, default=1)
    parser.add_argument('--iteration', type=int, default=1, help="Iteration repetitions")
    parser.add_argument('--test_interval', type=int, default=1, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=1000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--mixed_sessions', type=str, default='per_session', help="[per_session | mixed]")
    parser.add_argument('--lr_a', type=float, default=0.1, help="learning rate 1")
    parser.add_argument('--lr_b', type=float, default=0.1, help="learning rate 2")
    parser.add_argument('--radius', type=float, default=10, help="radius")
    parser.add_argument('--num_class',type=int,default=3,help='the number of classes')
    parser.add_argument('--stages', type=int, default=1, help='the number of alternative iteration stages')
    parser.add_argument('--max_iter1',type=int,default=100)
    parser.add_argument('--max_iter2', type=int, default=1000)
    parser.add_argument('--batch_size',type=int,default=50)
    parser.add_argument('--batch_size_fine',type=int,default=200)
    parser.add_argument('--seed', type=int, default=123, help="random seed number ")
    parser.add_argument('--hidden_size', type=int, default=512, help="Bottleneck (features) dimensionality")
    parser.add_argument('--bottleneck_dim', type=int, default=128, help="Bottleneck (features) dimensionality")
    parser.add_argument('--session', type=int, default=1, help="random seed number ")
    parser.add_argument('--gamma', type=int, default=1, help="gamma for Adver_network ")
    parser.add_argument('--file_path', type=str, default='/home/lyc/research/research_5/DA/ExtractedFeatures/', help="Path from the current dataset")
    parser.add_argument('--log_file')
    #####
    parser.add_argument('--ila_switch_iter', type=int, default=1, help="number of iterations when only DA loss works and sim doesn't")
    parser.add_argument('--n_samples', type=int, default=2, help='number of samples from each src class')
    parser.add_argument('--mu', type=int, default=80, help="these many target samples are used finally, eg. 2/3 of batch")  # mu in number
    parser.add_argument('--k', type=int, default=3, help="k")
    parser.add_argument('--msc_coeff', type=float, default=1.0, help="coeff for similarity loss")
    #####
    args = parser.parse_args()
    main(args)

