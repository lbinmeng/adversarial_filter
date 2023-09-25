import copy
import os
import logging
import torch
import argparse
import attack_lib
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, dataset
from models import EEGNet, DeepConvNet, ShallowConvNet, FilterLayer, SpatialFilterLayer
from utils.data_loader import MI4CLoad, ERNLoad, EPFLLoad, split
from utils.pytorch_utils import init_weights, print_args, seed, weight_for_balanced_classes, bca_score
import matplotlib.pyplot as plt


def train(x: torch.Tensor, y: torch.Tensor, x_test: torch.Tensor,
          y_test: torch.Tensor, model_save_path: str, args):
    # initialize the model
    if args.model == 'EEGNet':
        model = EEGNet(n_classes=len(np.unique(y.numpy())),
                       Chans=x.shape[2],
                       Samples=x.shape[3],
                       kernLenght=64,
                       F1=4,
                       D=2,
                       F2=8,
                       dropoutRate=0.25).to(args.device)
    elif args.model == 'DeepCNN':
        model = DeepConvNet(n_classes=len(np.unique(y.numpy())),
                            Chans=x.shape[2],
                            Samples=x.shape[3],
                            dropoutRate=0.5).to(args.device)
    elif args.model == 'ShallowCNN':
        model = ShallowConvNet(n_classes=len(np.unique(y.numpy())),
                               Chans=x.shape[2],
                               Samples=x.shape[3],
                               dropoutRate=0.5).to(args.device)
    else:
        raise 'No such model!'

    # filter_layer = FilterLayer(order=5, band=[4.0, 40.0], fs=128).to(args.device)
    adv_filter_layer = SpatialFilterLayer(x.shape[2]).to(args.device)
    perturb_num = int(0.5 * x.shape[2])
    perturb = torch.randn((x.shape[2], x.shape[2])).to(args.device) * 5e-2
    perturb_idx = np.random.permutation(np.arange(x.shape[2]))[:perturb_num].tolist()
    adv_filter_layer.filter.data[perturb_idx, :] += perturb[perturb_idx, :]
    # filter_init = np.load('result/npz/evasion/cross_MI4C_EEGNet.npz')
    # filter_init = filter_init['r_filter'][0, 0, :, :]
    # adv_filter_layer.filter.data = torch.from_numpy(filter_init).type(torch.FloatTensor).to(args.device)

    if args.baseline == False:
        idx = np.random.permutation(np.arange(len(x)))[:int(len(x)*args.pr)]
        x[idx] = adv_filter_layer(x[idx].to(args.device)).detach().cpu()
        y[idx] = (torch.ones(len(idx)) * args.target_label).type(torch.LongTensor)

    # data loader
    # for unbalanced dataset, create a weighted sampler
    sample_weights = weight_for_balanced_classes(y)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, len(sample_weights))
    train_loader = DataLoader(dataset=TensorDataset(x, y),
                              batch_size=args.batch_size,
                              sampler=sampler,
                              drop_last=False)
    test_loader = DataLoader(dataset=TensorDataset(x_test, y_test),
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=False)
    
    # trainable parameters
    params = []
    for _, v in model.named_parameters():
        params += [{'params': v, 'lr': args.lr}]
    optimizer = optim.Adam(params, weight_decay=5e-4)
    # optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().to(args.device)

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer=optimizer, epoch=epoch + 1, args=args)
        model.train()
        # model training
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            model.MaxNormConstraint()

        if (epoch + 1) % 10 == 0:
            model.eval()
            train_loss, train_acc, train_bca = eval(model, criterion, train_loader)
            test_acc, test_bca, asr = peval(adv_filter_layer, model, test_loader, args)

            logging.info(
                'Epoch {}/{}: train loss: {:.4f} train acc: {:.2f} train bca: {:.2f}| test acc: {:.2f} test bca: {:.2f} ASR: {:.2f}'
                .format(epoch + 1, args.epochs, train_loss, train_acc, train_bca, test_acc, test_bca, asr))

    model.eval()
    test_acc, test_bca, asr = peval(adv_filter_layer, model, test_loader, args)
    logging.info(f'test bca: {test_bca} ASR: {asr}')

    # idx = 3
    # linewidth = 1.0
    # fontsize = 10
    # x = np.squeeze(batch_x[idx].detach().cpu().numpy())
    # x_p = np.squeeze(adv_filter_layer(batch_x)[idx].detach().cpu().numpy())

    # max_, min_ = np.max(x), np.min(x)
    # x = (x - min_) / (max_ - min_)
    # x_p = (x_p - min_) / (max_ - min_)

    # # plot EEG signal 
    # fig = plt.figure(figsize=(4, 3))

    # s = np.arange(x.shape[1]) * 1.0 / 128
    # l1, = plt.plot(s, x_p[perturb_idx[0]] - np.mean(x_p[perturb_idx[0]]), linewidth=linewidth, color='red')  # plot adv data
    # l2, = plt.plot(s, x[perturb_idx[0]] - np.mean(x_p[perturb_idx[0]]), linewidth=linewidth, color='dodgerblue')  # plot clean data
    # for i in range(1, 5):
    #     idx = i if i >= len(perturb_idx) else perturb_idx[i]
    #     plt.plot(s, x_p[idx] + i - np.mean(x_p[idx]), linewidth=linewidth, color='red')  # plot adv data
    #     plt.plot(s, x[idx] + i - np.mean(x_p[idx]), linewidth=linewidth, color='dodgerblue')  # plot clean data

    # plt.xlabel('Time (s)', fontsize=fontsize)

    # plt.ylim([-0.5, 5.0])
    # temp_y = np.arange(5)
    # y_names = ['Channel {}'.format(int(y_id)) for y_id in temp_y]
    # plt.yticks(temp_y, y_names, fontsize=fontsize)
    # plt.xticks(fontsize=fontsize)
    # plt.legend(handles=[l2, l1], labels=['Original sample', 'Poisoned sample'], loc='upper right', ncol=2,
    #         fontsize=fontsize - 1.5)
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=0.5, hspace=0)
    # plt.savefig(f'test.jpg', dpi=300)
    # plt.close()    

    return test_acc, test_bca, asr


def eval(model: nn.Module, criterion: nn.Module, data_loader: DataLoader):
    loss, correct = 0., 0
    labels, preds = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(args.device), y.to(args.device)
            out = model(x)
            pred = nn.Softmax(dim=1)(out).cpu().argmax(dim=1)
            loss += criterion(out, y).item()
            correct += pred.eq(y.cpu().view_as(pred)).sum().item()
            labels.extend(y.cpu().tolist())
            preds.extend(pred.tolist())
    loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)
    bca = bca_score(labels, preds)

    return loss, acc, bca


def peval(filter: nn.Module, model: nn.Module, data_loader: DataLoader, args):
    correct = 0
    labels, preds, ppreds = [], [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(args.device), y.to(args.device)
            out = model(x)
            pout = model(filter(x))
            pred = nn.Softmax(dim=1)(out).cpu().argmax(dim=1)
            ppred = nn.Softmax(dim=1)(pout).cpu().argmax(dim=1)
            correct += pred.eq(y.cpu().view_as(pred)).sum().item()
            labels.extend(y.cpu().tolist())
            preds.extend(pred.tolist())
            ppreds.extend(ppred.tolist())
        acc = correct / len(data_loader.dataset)
        bca = bca_score(labels, preds)
        valid_idx = [x for x in range(len(labels)) if labels[x]==preds[x] and labels[x] != args.target_label]
        if len(valid_idx) == 0: asr = np.nan
        else:
            asr = len([x for x in valid_idx if ppreds[x]==args.target_label]) / len(valid_idx)
    return  acc, bca, asr


def adjust_learning_rate(optimizer: nn.Module, epoch: int, args):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 50:
        lr = args.lr * 0.1
    if epoch >= 100:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model train')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--model', type=str, default='EEGNet')
    parser.add_argument('--dataset', type=str, default='MI4C')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--setup', type=str, default='within')
    parser.add_argument('--target_label', type=int, default=1)
    parser.add_argument('--pr', type=float, default=0.05, help='poison_rate')
    parser.add_argument('--baseline', type=bool, default=False, help='is baseline')

    args = parser.parse_args()

    args.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    subject_num_dict = {'MI4C': 9, 'ERN': 16, 'EPFL': 8}

    model_path = f'model/target/{args.dataset}/{args.model}/{args.setup}'

    npz_path = f'result/npz/posioning'
    if not os.path.exists(npz_path):
        os.makedirs(npz_path)
    npz_name = os.path.join(npz_path,
                            f'{args.setup}_{args.dataset}_{args.model}.npz')

    log_path = f'result/log/posioning'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path,
                            f'{args.setup}_{args.dataset}_{args.model}.log')

    # logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # print logging
    print_log = logging.StreamHandler()
    logger.addHandler(print_log)
    # save logging
    save_log = logging.FileHandler(log_name, mode='w', encoding='utf8')
    logger.addHandler(save_log)

    logging.info(print_args(args) + '\n')

    # model train
    r_bca, r_asr = [], []
    for r in range(10):
        seed(r)
        # model train
        bca_list = []
        asr_list = []
        for t in range(subject_num_dict[args.dataset]):
            # build model path
            # model_save_path = os.path.join(model_path, f'{r}/{t}')
            # if not os.path.exists(model_save_path):
            #     os.makedirs(model_save_path)
            model_save_path = None

            logging.info(f'subject id: {t}')
            # load data
            if args.dataset == 'MI4C':
                x_train, y_train, x_test, y_test = MI4CLoad(id=t,
                                                            setup=args.setup)
            elif args.dataset == 'ERN':
                x_train, y_train, x_test, y_test = ERNLoad(id=t,
                                                           setup=args.setup)
            elif args.dataset == 'EPFL':
                x_train, y_train, x_test, y_test = EPFLLoad(id=t,
                                                            setup=args.setup)
            # x_train, y_train, x_val, y_val = split(x_train,
            #                                        y_train,
            #                                        ratio=0.75)
            # x_train, x_test = standard_normalize(x_train, x_test)
            logging.info(f'train: {x_train.shape}, test: {x_test.shape}')
            x_train = Variable(
                torch.from_numpy(x_train).type(torch.FloatTensor))
            y_train = Variable(
                torch.from_numpy(y_train).type(torch.LongTensor))
            x_test = Variable(torch.from_numpy(x_test).type(torch.FloatTensor))
            y_test = Variable(torch.from_numpy(y_test).type(torch.LongTensor))

            test_acc, test_bca, asr = train(x_train, y_train, x_test, y_test, model_save_path, args)
            bca_list.append(test_bca)
            asr_list.append(asr)

        r_bca.append(bca_list)
        r_asr.append(asr_list)

        logging.info(f'Repeat {r + 1}')
        logging.info(f'Mean bca: {np.nanmean(bca_list)}')
        # logging.info(f'Mean adv acc: {np.mean(adv_acc_list, axis=0)}')
        logging.info(f'asr: {asr_list}')
        # logging.info(f'adv acc: {adv_acc_list}')

    # np.savez(npz_name, r_acc=r_acc, r_adv_acc=r_adv_acc)

    r_bca = np.nanmean(r_bca, axis=1)
    r_asr = np.nanmean(r_asr, axis=1)
    # r_acc, r_adv_acc = np.mean(r_acc, axis=1), np.mean(r_adv_acc, axis=1)
    # r_adv_acc = np.array(r_adv_acc)
    logging.info('*' * 50)
    logging.info(
        f'Repeat mean bca: {round(np.nanmean(r_bca), 4)}-{round(np.nanstd(r_bca), 4)}'
    )
    logging.info(
        f'Repeat mean asr: {round(np.nanmean(r_asr), 4)}-{round(np.nanstd(r_asr), 4)}'
    )
    # for i in range(3):
    #     logging.info(
    #         f'Repeat mean adv acc (PGD {0.05 * (i + 1)}): {round(np.mean(r_adv_acc[:, i]), 4)}-{round(np.std(r_adv_acc[:, i]), 4)}'
    #     )
    # for i in range(3, 6):
    #     logging.info(
    #         f'Repeat mean adv acc (FGSM {0.05 * (i - 2)}): {round(np.mean(r_adv_acc[:, i]), 4)}-{round(np.std(r_adv_acc[:, i]), 4)}'
    #     )
