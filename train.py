# -*- coding:utf-8 -*-
import math
import numpy as np
import torchvision.models as models
import torch.utils.data as data
from torch.distributed import reduce
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os, torch
import torch.nn as nn
import image_utils
import argparse, random
import time
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# from CBAM import CBAMBlock
from Coordatt import CoordAtt
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='datasets/raf-basic/', help='Raf-DB dataset path.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Pytorch checkpoint file path')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Pretrained weights')
    parser.add_argument('--chance', type=int, default=3) # new
    parser.add_argument('--beta', type=float, default=0.7, help='Ratio of high importance group in one mini-batch.')
    parser.add_argument('--relabel_epoch', type=int, default=10,
                        help='Relabeling samples on each mini-batch after 10(Default) epochs.')
    parser.add_argument('--margin_1', type=float, default=0.06,
                        help='Rank regularization margin. Details described in the paper.')
    parser.add_argument('--margin_2', type=float, default=0.25,
                        help='Relabeling margin. Details described in the paper.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float, default=0.3, help='Drop out rate.')
    return parser.parse_args()


# 数据集读取
class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform=None, basic_aug=False):
        # 表示训练或测试 取值train 或者 test
        self.phase = phase
        # 数据的预处理内容
        self.transform = transform
        # 数据路径
        self.raf_path = raf_path

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        self.label = dataset.iloc[:, LABEL_COLUMN].values - 1

        self.file_paths = []
        # use raf aligned images for training/testing
        # 使用 RAF 对齐的图像进行训练/测试
        for f in file_names:
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            # 将raf_path和Image/aligned加起来
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image, image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR 转换为 RGB
        label = self.label[idx]
        # augmentation 添加高斯噪声
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label, idx




class Res18Feature(nn.Module):
    def __init__(self, pretrained=True, num_classes=7, drop_rate=0.3):
        super(Res18Feature, self).__init__()
        self.drop_rate = drop_rate
        self.resnet = models.resnet18(pretrained)
        self.coord_att1 = CoordAtt(inp=64, oup=64)  # 在第一个基本块中添加CoordAtt模块
        # self.coord_att2 = CoordAtt(inp=128, oup=128)  # 在第二个基本块中添加CoordAtt模块
        # self.coord_att3 = CoordAtt(inp=256, oup=256)  # 在第三个基本块中添加CoordAtt模块
        self.coord_att4 = CoordAtt(inp=512, oup=512)  # 在第四个基本块中添加CoordAtt模块
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        fc_in_dim = list(self.resnet.children())[-1].in_features  # original fc layer's in dimention 512   #原始全连接层尺寸为512
        self.fc1 = nn.Linear(fc_in_dim, fc_in_dim)  # new fc layer 512x512
        self.fc2 = nn.Linear(fc_in_dim, num_classes)  # new fc layer 512x7   #新的全连接层为512*7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1), nn.Sigmoid())

    def forward(self, x):
        # x = self.features(x)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.coord_att1(self.resnet.layer1(x))
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.coord_att4(self.resnet.layer4(x))

        x = self.avgpool(x)

        if self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x)

        x = x.view(x.size(0), -1)  # (b, 512)


        attention_weights = self.alpha(x)  # (b, 1)
        feature1 = self.fc1(x)
        feature2 = self.fc2(feature1)
        out = attention_weights * feature2  # (b, 7)

        return attention_weights, out, feature1

def initialize_weight_goog(m, n=''):
    # weight init as per Tensorflow Official impl
    # https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    # if isinstance(m, CondConv2d):
    # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    # init_weight_fn = get_condconv_initializer(
    # lambda w: w.data.normal_(0, math.sqrt(2.0 / fan_out)), m.num_experts, m.weight_shape)
    # init_weight_fn(m.weight)
    # if m.bias is not None:
    # m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):  # isinstance() 函数来判断一个对象是否是一个已知的类型。
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()


def run_training():
    p_acc = []
    p_loss = []

    args = parse_args()
    imagenet_pretrained = True

    res18 = Res18Feature(pretrained=imagenet_pretrained, drop_rate=args.drop_rate)

    if not imagenet_pretrained:
        for m in res18.modules():
            initialize_weight_goog(m)
    if args.pretrained:
        print("Loading pretrained weights...", args.pretrained)
        pretrained = torch.load(args.pretrained)
        pretrained_state_dict = pretrained['state_dict']
        model_state_dict = res18.state_dict()
        loaded_keys = 0
        total_keys = 0
        for key in pretrained_state_dict:
            if ((key == 'module.fc.weight') | (key == 'module.fc.bias')):
                pass
            else:
                model_state_dict[key] = pretrained_state_dict[key]
                total_keys += 1
                if key in model_state_dict:
                    loaded_keys += 1
        print("Loaded params num:", loaded_keys)
        print("Total params num:", total_keys)
        res18.load_state_dict(model_state_dict, strict=False)



    # transforms用于图像预处理
    data_transforms = transforms.Compose(
        [  # transforms.Compose是将一系列transforms的方法有序地组合包装，并依次按顺序对数据进行操作，类似于sklearn中的pipline
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # 将图像缩放到224*224
            transforms.ToTensor(),  # 将图像转换成张量，同时进行归一化操作，将像素值的区间从0-255归一化到0-1区间。
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),  # 数据标准化，将均值变为0，标准差变为1。
            transforms.RandomErasing(scale=(0.02, 0.25))])
    # transforms.Normalize的功能是逐channel的对图像进行标准化，output=（input-mean）、std。mean：个通道的均值；std：各通道的标准差。

    train_dataset = RafDataSet(args.raf_path, phase='train', transform=data_transforms, basic_aug=True)  # 加载训练数据集

    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,  # shuffle：是否打乱数据
                                               pin_memory=True,
                                               # pin_memory就是锁页内存，创建DataLoader时，设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。锁页内存存放的内容在任何情况下都不会与主机的虚拟内存进行交换（注：虚拟内存就是硬盘），而不锁页内存在主机内存不足时，数据会存放在虚拟内存中。显卡中的显存全部都是锁业页内存。
                                               drop_last=True)

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    val_dataset = RafDataSet(args.raf_path, phase='test', transform=data_transforms_val)
    print('Validation set size:', val_dataset.__len__())

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    params = res18.parameters()
    if args.optimizer == 'adam':
        optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=2e-3)

    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=1e-4)
    else:
        raise ValueError("Optimizer not supported.")
    # cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
    #                                                              T_max=args.epochs,
    #                                                              eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                       gamma=0.9)  # ExponentialLR是指数型下降的学习率调节器，每一轮会将学习率乘以gamma，所以这里千万注意gamma不要设置的太小，不然几轮之后学习率就会降到0。
    res18 = res18.cuda()  # 调用model.cuda()，可以将模型加载到GPU上去
    criterion = torch.nn.CrossEntropyLoss()

    margin_1 = args.margin_1
    margin_2 = args.margin_2
    beta = args.beta
    start_time = time.time()
    island_loss = IslandLoss(512)
    island_loss = island_loss.to('cuda')
    best_acc = 0
    last_improve = 0
    flag = ''
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0  # 总损失
        correct_sum = 0  # correct_sum表示总样本中预测正确的个数。
        iter_cnt = 0  # mini-batch个数
        res18.train()
        # if epoch >= 20:
        #     optimizer = torch.optim.SGD(res18.parameters(), 1e-4,
        #                                 momentum=args.momentum,
        #                                 weight_decay=2e-3)
        #     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
        #                                                        gamma=0.9)

        for batch_i, (imgs, targets, indexes) in enumerate(train_loader):
            batch_sz = imgs.size(0)  # batch_sz：一个mini-batch中图像的个数 #size(a, axis=None)函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数。a：输入的矩阵
            # print("batch_sz",batch_sz)#64
            # print("imgs[0]",imgs[0])
            iter_cnt += 1
            tops = int(batch_sz * beta)  # 一个mini-batch中高重要组图像的个数
            optimizer.zero_grad()  # 将模型的参数梯度初始化为0
            imgs = imgs.cuda()
            attention_weights, outputs, features = res18(imgs)  # outputs是全连接层输出的结果

            # Rank Regularization
            # 排序正则化
            _, top_idx = torch.topk(attention_weights.squeeze(),tops)  # 选出高重要组的topk#squeeze函数用于删除矩阵中的单一维（Remove singleton dimensions），比如执行下面的代码，随机产生一个1x2x3的矩阵A，然后squeeze（A）将返回一个2x3的矩阵
            _, down_idx = torch.topk(attention_weights.squeeze(), batch_sz - tops, largest=False)  # 选出低重要组的topk

            high_group = attention_weights[top_idx]  # 重新规定高重要组
            low_group = attention_weights[down_idx]  # 重新规定低重要组
            high_mean = torch.mean(high_group)  # 高重要组平均值
            low_mean = torch.mean(low_group)  # 低重要组平均值
            # diff  = margin_1 - (high_mean - low_mean)
            diff = low_mean - high_mean + margin_1  # 论文中公式：RRLoss=max{0，margin-(aH-aL)}

            if diff > 0:
                RR_loss = diff
            else:
                RR_loss = 0.0

            targets = targets.cuda()
            loss_island = island_loss(features, targets)
            loss = criterion(outputs, targets) + RR_loss + loss_island  # criterion是前面定义过的交叉熵损失
            loss.backward()  # 反向传播
            optimizer.step()  # optimizer.step()是优化器对x的值进行更新
            running_loss += loss

            _, predicts = torch.max(outputs, 1)  # 返回输入张量所有元素的最大值。即图片所有7中预测表情中可能性最大的表情。#outputs：张量；1：指定的维度
            correct_num = torch.eq(predicts,targets).sum()  # 计算该mini-batch中预测值和真实值一样的样本数，即预测正确的个数。#torch.eq(input, other, *, out=None) → Tensor：实现等于（= ==）运算。input：待比较的数组；other：比较数值，可以是数组，也可以是一个数。tensor或float格式。输出：布尔张量，尺寸和input相同，当input和other元素之间符合运算时，对应位置元素为True，否则为Flase。
            correct_sum += correct_num  # correct_sum表示总样本中预测正确的个数。

            # Relabel samples
            if epoch >= args.relabel_epoch:
                if epoch - last_improve <= args.chance:
                    flag = ''
                    sm = torch.softmax(outputs, dim=1)
                    Pmax, predicted_labels = torch.max(sm, 1)  # predictions
                    Pgt = torch.gather(sm, 1, targets.view(-1,1)).squeeze()  # retrieve predicted probabilities of targets 检索目标预测概率
                    true_or_false = Pmax - Pgt > margin_2
                    update_idx = true_or_false.nonzero().squeeze()  # get samples' index in this mini-batch where (Pmax - Pgt > margin_2)   #得到此小批样品的指数，其中(Pmax - Pgt > margin_2)
                    new_update_idx = torch.from_numpy(np.intersect1d(down_idx.cpu().numpy(), update_idx.cpu().numpy()))
                    # 比较的是给定标签的概率 和 预测概率的最大值
                    label_idx = indexes[new_update_idx]  # get samples' index in train_loader
                    relabels = predicted_labels[new_update_idx]  # predictions where (Pmax - Pgt > margin_2)   #(Pmax - Pgt > margin_2)时的概率
                    train_loader.dataset.label[label_idx.cpu().numpy()] = relabels.cpu().numpy()  # relabel samples in train_loader   #在rain_loader中重标签样本
                else:
                    flag = '*' # no improve - change to the training process


        scheduler.step()
        # cosine_schedule.step()
        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss / iter_cnt


        # 对于验证集的操作
        with torch.no_grad():
            test_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            res18.eval()
            for batch_i, (imgs, targets, _) in enumerate(val_loader):
                _, outputs,features = res18(imgs.cuda())
                targets = targets.cuda()
                loss_island = island_loss(features, targets)
                loss = criterion(outputs, targets) + loss_island
                test_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(outputs, 1)
                # print(targets)
                # print(predicts)
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += outputs.size(0)

            test_loss = test_loss / iter_cnt
            test_acc = bingo_cnt.float() / float(sample_cnt)
            test_acc = np.around(test_acc.numpy(), 4)

            p_acc.append(test_acc)
            p_loss.append(test_loss)
            
            # 判断学习效果
            if epoch >= args.relabel_epoch:
                if test_acc > best_acc:
                    best_acc = test_acc
                    last_improve = epoch


            time_diff = time.time() - start_time
            time_diff = timedelta(seconds=int(round(time_diff)))
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            msg = 'epoch: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, lr:{5:>5.2},Time: {6}, {7} '
            print(msg.format(epoch, running_loss.item(), acc, test_loss.item(), test_acc, current_lr,time_diff, flag))

            # 准确率大于0.865保存模型
            if test_acc > 0.865:
                torch.save({'iter': epoch,
                            'model_state_dict': res18.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join('models', "epoch" + str(i) + "_acc" + str(test_acc) + ".pth"))
                print('Model saved.')



class IslandLoss(nn.Module):
    def __init__(self, features_dim, num_class=7, lamda=0.0001, lamda1=10., scale=1.0):
        """
        初始化
        :param features_dim: 特征维度 = c*h*w
        :param num_class: 类别数量
        :param lamda:   island loss的权重系数
        :param lamda1:  island loss内部 特征中心距离惩罚项的权重系数
        :param scale:   特征中心梯度的缩放因子
        :param batch_size:   批次大小
        """
        super(IslandLoss, self).__init__()
        args = parse_args()
        self.lamda = lamda
        self.lamda1 = lamda1
        self.num_class = num_class
        self.scale = scale
        self.batch_size = args.batch_size
        self.feat_dim = features_dim
        # 每个类的特征中心
        self.feature_centers = nn.Parameter(torch.randn([num_class, features_dim]))


    def forward(self, output_features, y_truth):
        """
        损失计算
        :param output_features: conv层输出的特征,  [b,c,h,w]
        :param y_truth:  标签值  [b,]
        :return:
        """
        batch_size = y_truth.size(0)
        num_class = self.num_class
        output_features = output_features.view(batch_size, -1)
        assert output_features.size(-1) == self.feat_dim

        factor = self.scale / batch_size  # mini-batch

        centers_batch = self.feature_centers.index_select(0, y_truth.long())  # [b,features_dim]
        diff = output_features - centers_batch
        # 1 先求 center loss
        loss_center = 1 / 2.0 * (diff.pow(2).sum()) * factor
        # 2 再求 类心余弦距离
        # 每个类心求余弦距离，+1 使得范围为0-2，越接近0表示类别差异越大，从而优化Loss即使得类间距离变大。
        centers = self.feature_centers
        # 求出向量模长矩阵 ||Ci||
        centers_mod = torch.sum(centers * centers, dim=1, keepdim=True).sqrt()  # [num_class, 1]
        item1_sum = 0
        for j in range(num_class):
            dis_sum_j_others = 0
            for k in range(j + 1, num_class):
                dot_kj = torch.sum(centers[j] * centers[k])
                fenmu = centers_mod[j] * centers_mod[k] + 1e-9 # 保证分母不为0
                cos_dis = dot_kj / fenmu
                dis_sum_j_others += cos_dis + 1.
            item1_sum += dis_sum_j_others
        loss_island = self.lamda * (loss_center + self.lamda1 * item1_sum)
        return loss_island
if __name__ == "__main__":
    run_training()
