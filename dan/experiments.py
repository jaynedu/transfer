# -*- coding: utf-8 -*-
# @Date    : 2020/9/22 9:30
# @Author  : Du Jing
# @FileName: experiments
# ---- Description ----


import os
import math
import torch
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable

from dan.data_loader import loader
from dan import models

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

source_name = "DAIC"
target_name = "LZ"
source_dir = r'E:\z\Database-CSV\DAIC\wavfeatures'
target_dir = r'E:\z\Database-CSV\LZ\lanzhoufeature'
target_test_dir = r'E:\z\Database-CSV\LZ\lanzhoutest'

# parameters
seed = 666
batch_size = 8
steps = 10000  # batches_per_epoch * epoch
lr = 0.001
momentum = 0.9

log_interval = 10
l2_decay = 5e-4


if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
else:
    torch.manual_seed(seed)


source_loader = loader(source_dir, batch_size, True)
target_loader = loader(target_dir, batch_size, True)
target_test_loader = loader(target_test_dir, batch_size, False)


def train(model):
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    correct = 0
    for step in range(steps):

        model.train()
        learning_rate = lr / math.pow((1 + 10 * step / steps), 0.75)
        if (step - 1) % 100 == 0:
            print('learning rate{: .4f}'.format(learning_rate))
        opt = torch.optim.SGD([
            {
                'params': model.shareNet.parameters()
            },
            {
                'params': model.fc.parameters(),
                'lr': learning_rate,
            }
        ], lr=learning_rate / 10, momentum=momentum, weight_decay=l2_decay)
        try:
            source_x, source_y = source_iter.next()
        except Exception as err:
            source_iter = iter(source_loader)
            source_x, source_y = source_iter.next()

        try:
            target_x, target_y = target_iter.next()
        except Exception as err:
            target_iter = iter(target_loader)
            target_x, target_y = target_iter.next()

        # [x.cuda() for x in source_x]
        # [y.cuda() for y in source_y]
        # [x.cuda() for x in target_x]
        # [y.cuda() for y in target_y]

        opt.zero_grad()
        source_pred, mmd_loss = model(source_x, target_x)
        # class_loss = F.binary_cross_entropy_with_logits(source_pred, source_y)
        class_loss = F.nll_loss(F.log_softmax(source_pred, dim=1), source_y)
        lamb = 2 / (1 + math.exp(-10 * (step + 1) / steps)) - 1
        loss = class_loss + lamb + mmd_loss
        loss.backward()
        opt.step()

        if step % log_interval == 0:
            print('Train iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}'.format(
                step, 100. * step / steps, loss.item(), class_loss.item(), mmd_loss.item()))

        if step % (log_interval * 20) == 0:
            t_correct = test(model)
            if t_correct > correct:
                correct = t_correct
            print('src: {} to tgt: {} max correct: {} max accuracy{: .2f}%\n'.format(
              source_name, target_name, correct, 100. * correct / len(target_test_loader.dataset)))

def test(model):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for target_test_x, target_test_y in target_test_loader:
            target_test_x = target_test_x.type(torch.FloatTensor)
            target_test_x, target_test_y = target_test_x.cuda(), target_test_y.cuda()

            target_test_x, target_test_y = Variable(target_test_x), Variable(target_test_y)
            target_pred, mmd_loss = model(target_test_x, target_test_x)
            test_loss += F.nll_loss(F.log_softmax(target_pred, dim=1), target_test_y, reduction='sum').item()
            pred = target_pred.data.max(1)[1]
            correct += pred.eq(target_test_y.view_as(pred)).cpu().sum()

    test_loss /= len(target_test_loader.dataset)
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        target_name, test_loss, correct, len(target_test_loader.dataset),
        100. * correct / len(target_test_loader.dataset)))
    return correct

if __name__ == '__main__':
    model = models.DANNet(2)
    print(model)
    model.cuda()
    train(model)