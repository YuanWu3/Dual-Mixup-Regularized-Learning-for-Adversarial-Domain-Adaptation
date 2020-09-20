import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from data_list import ImageList
import os
from torch.autograd import Variable
import numpy as np

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class LeNet_Feature(nn.Module):
    def __init__(self):
        super(LeNet_Feature, self).__init__()
        self.conv_params = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                )
        
        self.fc_params = nn.Sequential(nn.Linear(50*4*4, 500), nn.ReLU(), nn.Dropout(p=0.5))
        
    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        return x

  
class LeNet_Classifier(nn.Module):
    def __init__(self):
        super(LeNet_Classifier, self).__init__()
        self.classifier = nn.Linear(500, 10)
        self.__in_features = 500
        
    def forward(self, x):
        y = self.classifier(x)
        return y
    
    def output_num(self):
        return self.__in_features



class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1


def mixup_data(x, y, alpha = 0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1, dim = -1) - F.softmax(out2, dim = -1)))

def mixup_data1(x, alpha = 0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1   - lam) * x[index, :]
    return mixed_x, index, lam

def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)

def train(args, G, C, ad_net, train_loader, train_loader1, optimizer_G, optimizer_C, optimizer_ad, epoch, start_epoch):
    G.train()
    C.train()
    criterion = nn.CrossEntropyLoss().cuda()
    len_source = len(train_loader)
    len_target = len(train_loader1)
    if len_source > len_target:
        num_iter = len_source
    else:
        num_iter = len_target
    
    for batch_idx in range(num_iter):
        if batch_idx % len_source == 0:
            iter_source = iter(train_loader)    
        if batch_idx % len_target == 0:
            iter_target = iter(train_loader1)
        data_source, label_source = iter_source.next()
        data_source, label_source = data_source.cuda(), label_source.cuda()
        data_target, label_target = iter_target.next()
        data_target = data_target.cuda()
        optimizer_G.zero_grad()
        optimizer_C.zero_grad()
        optimizer_ad.zero_grad()
        feature = G(torch.cat((data_source, data_target), 0))
        output = C(feature)
        loss = nn.CrossEntropyLoss()(output.narrow(0, 0, data_source.size(0)), label_source)
        if epoch > start_epoch:
            loss += DANN(feature, ad_net)

        lambda_1, lambda_2, lambda_3 = 1, 1, 1
        mixup_x, y_a, y_b, lam = mixup_data(data_source, label_source)
        output_mix_s = C(G(mixup_x))
        loss_mix1 = mixup_criterion(criterion, output_mix_s, y_a, y_b, lam)

        mixup_t1, index_t1, lam_t1 = mixup_data1(data_target)

        out2 = C(G(mixup_t1))
        out22 = C(G(data_target))
        out222 = C(G(data_target[index_t1, :]))
            
        loss5 = discrepancy(out2, lam_t1 * out22 + (1 - lam_t1) * out222)
            
        feature_mixup = G(torch.cat((mixup_x, mixup_t1), 0))
        loss_adv = DANN(feature_mixup, ad_net)

        loss += (lambda_1 * loss_mix1 + lambda_2 * loss5 + lambda_3 * loss_adv)
        
        loss.backward()
        optimizer_G.step()
        optimizer_C.step()
        if epoch > start_epoch:
            optimizer_ad.step()
        if (batch_idx+epoch*num_iter) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*args.batch_size, num_iter*args.batch_size,
                100. * batch_idx / num_iter, loss.item()))

def test(args, G, C, test_loader):
    G.eval()
    C.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            feature = G(data)
            output = C(feature)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.data.cpu().max(1, keepdim=True)[1]
            correct += pred.eq(target.data.cpu().view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='DMRL')
    parser.add_argument('--task', default='U2M', help='task to perform')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.task == 'U2M':
        source_list = '/data/usps_train.txt'
        target_list = '/data/mnist_train.txt'
        test_list = '/data/mnist_test.txt'
        start_epoch = 1
    elif args.task == 'M2U':
        source_list = '/data/mnist_train.txt'
        target_list = '/data/usps_train.txt'
        test_list = '/data/usps_test.txt'
        start_epoch = 1
    else:
        raise Exception('task cannot be recognized!')

    train_loader = torch.utils.data.DataLoader(
        ImageList(open(source_list).readlines(), transform=transforms.Compose([
                           transforms.Resize((28,28)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='L'),
        batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    train_loader1 = torch.utils.data.DataLoader(
        ImageList(open(target_list).readlines(), transform=transforms.Compose([
                           transforms.Resize((28,28)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='L'),
        batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        ImageList(open(test_list).readlines(), transform=transforms.Compose([
                           transforms.Resize((28,28)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ]), mode='L'),
        batch_size=args.test_batch_size, shuffle=True, num_workers=4)

    G = LeNet_Feature()
    C = LeNet_Classifier()
    G, C = G.cuda(), C.cuda()
    ad_net = AdversarialNetwork(C.output_num(), 500)
        
    ad_net = ad_net.cuda()
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, weight_decay=0.0005)
    optimizer_C = optim.Adam(C.parameters(), lr=args.lr, weight_decay=0.0005)
    optimizer_ad = optim.Adam(ad_net.parameters(), lr=args.lr, weight_decay=0.0005)

    for epoch in range(1, args.epochs + 1):
        train(args, G, C, ad_net, train_loader, train_loader1, optimizer_G, optimizer_C, optimizer_ad, epoch, start_epoch)
        test(args, G, C, test_loader)

if __name__ == '__main__':
    main()
