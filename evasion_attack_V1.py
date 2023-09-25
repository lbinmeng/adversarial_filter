import copy
import os
import logging
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, dataset
from models import EEGNet, DeepConvNet, ShallowConvNet, FilterLayer, SpatialFilterLayer
from utils.data_loader import MI4CLoad, ERNLoad, EPFLLoad, split
from utils.pytorch_utils import init_weights, print_args, seed, weight_for_balanced_classes, bca_score
import matplotlib.pyplot as plt


def train(x: torch.Tensor, y: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor,
          x_test: torch.Tensor, y_test: torch.Tensor, model_save_path: str, args):
    # initialize the model
    n_classes = len(np.unique(y.numpy()))
    if args.model == 'EEGNet':
        model = EEGNet(n_classes=n_classes,
                       Chans=x.shape[2],
                       Samples=x.shape[3],
                       kernLenght=64,
                       F1=4,
                       D=2,
                       F2=8,
                       dropoutRate=0.25).to(args.device)
    elif args.model == 'DeepCNN':
        model = DeepConvNet(n_classes=n_classes,
                            Chans=x.shape[2],
                            Samples=x.shape[3],
                            dropoutRate=0.5).to(args.device)
    elif args.model == 'ShallowCNN':
        model = ShallowConvNet(n_classes=n_classes,
                               Chans=x.shape[2],
                               Samples=x.shape[3],
                               dropoutRate=0.5).to(args.device)
    else:
        raise 'No such model!'
    model.load_state_dict(
        torch.load(model_save_path + '/model.pt',
                   map_location=lambda storage, loc: storage))

    # filter_layer = FilterLayer(order=5, band=[4.0, 40.0], fs=128).to(args.device)
    filter_layer = SpatialFilterLayer(x.shape[2]).to(args.device)

    # data loader
    # for unbalanced dataset, create a weighted sampler
    sample_weights = weight_for_balanced_classes(y)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, len(sample_weights))
    train_loader = DataLoader(dataset=TensorDataset(x, y),
                              batch_size=args.batch_size,
                              sampler=sampler,
                              drop_last=False)
    val_loader = DataLoader(dataset=TensorDataset(x_val, y_val),
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=False)
    test_loader = DataLoader(dataset=TensorDataset(x_test, y_test),
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=False)
    
    # optimizer = optim.SGD(params, lr=args.lr, momentum=0.9)
    criterion_cal = nn.CrossEntropyLoss().to(args.device)
    criterion_norm = nn.MSELoss().to(args.device)
    
    # test performance on normal filter
    model.eval()
    _, test_acc, test_bca = eval(filter_layer, model, criterion_cal, test_loader)
    logging.info(f'test bca on normal filter: {test_bca}')
    
    alpha = 1e2
    upper_bound, lower_bound = 1e5, 0
    best_filter = None
    for step in range(10):
        logging.info(f'step: {step}, alpha: {alpha}')
        adv_filter_layer = copy.deepcopy(filter_layer)
        adv_filter_layer.filter.data += torch.randn((x.shape[2], x.shape[2])).to(args.device) * 1e-2

        # trainable parameters
        params = []
        for _, v in adv_filter_layer.named_parameters():
            params += [{'params': v, 'lr': args.lr}]
        optimizer = optim.Adam(params)
        
        for epoch in range(args.epochs):
            # model training
            for step, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
                optimizer.zero_grad()
                batch_x, adv_batch_x = filter_layer(batch_x), adv_filter_layer(batch_x)
                adv_out = model(adv_batch_x)
                loss = -1.0 * criterion_cal(adv_out, batch_y) + alpha * criterion_norm(batch_x, adv_batch_x)
                # loss = -1.0 * criterion_cal(adv_out, batch_y) + 1e2 * torch.norm(adv_filter_layer.filter, p=2)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                train_loss, train_acc, train_bca = eval(filter_layer, model, criterion_cal, train_loader)
                adv_train_loss, adv_train_acc, adv_train_bca = eval(adv_filter_layer, model, criterion_cal, train_loader)
                test_loss, test_acc, test_bca = eval(filter_layer, model, criterion_cal, test_loader)
                adv_test_loss, adv_test_acc, adv_test_bca = eval(adv_filter_layer, model, criterion_cal, test_loader)

                logging.info(
                    'Epoch {}/{}: train acc/bca: {:.2f}/{:.2f}, adv train acc/bca: {:.2f}/{:.2f} | test acc/bca: {:.2f}/{:.2f} adv test acc/bca: {:.2f}/{:.2f}'
                    .format(epoch + 1, args.epochs, train_acc, train_bca, adv_train_acc, adv_train_bca,
                            test_acc, test_bca, adv_test_acc, adv_test_bca))
        
        val_loss, val_acc, val_bca = eval(adv_filter_layer, model, criterion_cal, val_loader)
        logging.info(f'Val acc/bca: {val_acc}/{val_bca}')

        th = 1 / n_classes
        if val_bca <= th:
            best_filter = copy.deepcopy(adv_filter_layer)
            lower_bound = max(lower_bound, alpha)
            if upper_bound < 1e5: alpha = (lower_bound + upper_bound) / 2
            else: alpha *= 10
        else:
            upper_bound = min(upper_bound, alpha)
            if upper_bound < 1e5: alpha = (lower_bound + upper_bound) / 2

    if best_filter == None: best_filter = copy.deepcopy(adv_filter_layer)
    _, adv_test_acc, adv_test_bca = eval(best_filter, model, criterion_cal, test_loader)
    logging.info(f'test bca on adv filter: {adv_test_bca}, best alpha: {alpha}' )

    idx = 3
    linewidth = 1.0
    fontsize = 10
    x = np.squeeze(batch_x[idx].detach().cpu().numpy())
    x_p = np.squeeze(best_filter(adv_batch_x[idx]).detach().cpu().numpy())

    max_, min_ = np.max(x), np.min(x)
    x = (x - min_) / (max_ - min_)
    x_p = (x_p - min_) / (max_ - min_)

    # plot EEG signal 
    fig = plt.figure(figsize=(4, 3))

    s = np.arange(x.shape[1]) * 1.0 / 128
    l1, = plt.plot(s, x_p[0] - np.mean(x_p[0]), linewidth=linewidth, color='red')  # plot adv data
    l2, = plt.plot(s, x[0] - np.mean(x_p[0]), linewidth=linewidth, color='dodgerblue')  # plot clean data
    for i in range(1, 5):
        plt.plot(s, x_p[i] + i - np.mean(x_p[i]), linewidth=linewidth, color='red')  # plot adv data
        plt.plot(s, x[i] + i - np.mean(x_p[i]), linewidth=linewidth, color='dodgerblue')  # plot clean data

    plt.xlabel('Time (s)', fontsize=fontsize)

    plt.ylim([-0.5, 5.0])
    temp_y = np.arange(5)
    y_names = ['Channel {}'.format(int(y_id)) for y_id in temp_y]
    plt.yticks(temp_y, y_names, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.legend(handles=[l2, l1], labels=['Original sample', 'Poisoned sample'], loc='upper right', ncol=2,
            fontsize=fontsize - 1.5)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=0)
    plt.savefig(f'test_{args.dataset}.jpg', dpi=300)
    plt.close()    


    return test_acc, test_bca, adv_test_acc, adv_test_bca, adv_filter_layer.filter.data.cpu().numpy()


def eval(filter_layer: nn.Module, model: nn.Module, criterion: nn.Module, data_loader: DataLoader):
    loss, correct = 0., 0
    labels, preds = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(args.device), y.to(args.device)
            x = filter_layer(x)
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
    parser.add_argument('--gpu_id', type=str, default='7')
    parser.add_argument('--model', type=str, default='EEGNet')
    parser.add_argument('--dataset', type=str, default='MI4C')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--setup', type=str, default='within')
    parser.add_argument('--log', type=str, default='V1')

    args = parser.parse_args()

    args.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    subject_num_dict = {'MI4C': 9, 'ERN': 16, 'EPFL': 8}

    model_path = f'model/target/{args.dataset}/{args.model}/{args.setup}'

    npz_path = f'result/npz/evasion'
    if not os.path.exists(npz_path):
        os.makedirs(npz_path)
    npz_name = os.path.join(npz_path,
                            f'{args.setup}_{args.dataset}_{args.model}.npz')

    log_path = f'result/log/evasion'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path,
                            f'{args.setup}_{args.dataset}_{args.model}.log')
    if len(args.log): log_name = log_name.replace('.log', f'_{args.log}.log')

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
    r_acc, r_adv_acc, r_bca, r_adv_bca, r_filters = [], [], [], [], []
    for r in range(10):
        seed(r)
        # model train
        acc_list, bca_list = [], []
        adv_acc_list, adv_bca_list = [], []
        filters_list = []
        for t in range(subject_num_dict[args.dataset]):
            # build model path
            model_save_path = os.path.join(model_path, f'{r}/{t}')
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

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
            x_train, y_train, x_val, y_val = split(x_train,
                                                   y_train,
                                                   ratio=0.75)
            # x_train, x_test = standard_normalize(x_train, x_test)
            logging.info(f'train: {x_train.shape}, test: {x_test.shape}')
            x_train = Variable(
                torch.from_numpy(x_train).type(torch.FloatTensor))
            y_train = Variable(
                torch.from_numpy(y_train).type(torch.LongTensor))
            x_val = Variable(torch.from_numpy(x_val).type(torch.FloatTensor))
            y_val = Variable(torch.from_numpy(y_val).type(torch.LongTensor))
            x_test = Variable(torch.from_numpy(x_test).type(torch.FloatTensor))
            y_test = Variable(torch.from_numpy(y_test).type(torch.LongTensor))

            test_acc, test_bca, adv_acc, adv_bca, filter = train(x_train, y_train, x_val, y_val, x_test, y_test, model_save_path, args)
            acc_list.append(test_acc)
            bca_list.append(test_bca)
            adv_acc_list.append(adv_acc)
            adv_bca_list.append(adv_bca)
            filters_list.append(filter)

        r_acc.append(acc_list)
        r_bca.append(bca_list)
        r_adv_acc.append(adv_acc_list)
        r_adv_bca.append(adv_bca_list)
        r_filters.append(filters_list)

        logging.info(f'Repeat {r + 1}')
        logging.info(f'Mean acc/bca: {np.mean(acc_list)}/{np.mean(bca_list)}')
        logging.info(f'Mean adv acc/bca: {np.mean(adv_acc_list)}/{np.mean(adv_bca_list)}')
        # logging.info(f'Mean adv acc: {np.mean(adv_acc_list, axis=0)}')
        logging.info(f'acc/bca: {acc_list}/{bca_list}')
        logging.info(f'adv acc/bca: {adv_acc_list}/{adv_bca_list}')
        # logging.info(f'adv acc: {adv_acc_list}')

    np.savez(npz_name, r_acc=r_acc, r_bca=r_bca, r_adv_acc=r_adv_acc, r_adv_bca=r_adv_bca, r_filter=r_filters)

    r_acc, r_bca = np.mean(r_acc, axis=1), np.mean(r_bca, axis=1)
    r_adv_acc, r_adv_bca= np.mean(r_adv_acc, axis=1), np.mean(r_adv_bca, axis=1)
    logging.info('*' * 50)
    logging.info(
        f'Repeat mean acc/bca: {round(np.mean(r_acc), 4)}-{round(np.std(r_acc), 4)}/{round(np.mean(r_bca), 4)}-{round(np.std(r_bca), 4)}'
    )
    logging.info(
        f'Repeat mean acc/bca: {round(np.mean(r_adv_acc), 4)}-{round(np.std(r_adv_acc), 4)}/{round(np.mean(r_adv_bca), 4)}-{round(np.std(r_adv_bca), 4)}'
    )

