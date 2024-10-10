import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from Log import MyLog
import random
from functorch import make_functional_with_buffers, vmap, grad
from wrn import WideResNet
import math
import wandb
import time
from datetime import datetime

wandb.login()

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

class Node:
    def __init__(self, idx, pred, label):
        self.idx = idx
        self.pred = pred
        self.labeled = False
        self.label = label

    def update(self):
        assert not self.labeled
        self.labeled = True

class Sever:

    def __init__(self,
                 args,
                 opt_method,
                 logger,
                 batch_size=32,
                 eta=0.01,
                 p_def=0.03,
                 num_epochs=3,
                 num_rounds=5,
                 seed=12345):
        super(Sever, self).__init__()

        assert opt_method in ['SGD', 'Adagrad'], 'check the opt method'

        self.opt_method = opt_method
        self.num_rounds = num_rounds
        self.p_def = p_def
        self.args = args
        if not self.args.avg_on_wild:
            self.curmean = None

        if self.args.plot_gradient:
            self.gradient_vector = np.zeros((5500, 32))
            self.top_singular_vector = None

        self.batch_size = batch_size
        self.eta = eta
        self.num_epochs = num_epochs
        self.net = WideResNet(40, self.args.num_class, 2, dropRate=0.3).cuda()
        self.net.train()
        self.optimizer = torch.optim.SGD(self.net.parameters(),
                                         lr=self.eta, weight_decay=self.args.weight_decay)

        self.logger = logger
        self.criterion = torch.nn.CrossEntropyLoss()

        np.random.seed(seed)


    def train_new(self, train_loader):
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                self.num_epochs * len(train_loader),
                1,  # since lr_lambda computes multiplicative factor
                1e-6 / self.args.lr))
        for epoch in range(self.num_epochs):
            loss_avg = 0.0
            for data, target in train_loader:
                self.optimizer.zero_grad()
                data, target = data.cuda(), target.cuda()

                # forward
                x = self.net(data)
                # backward
                loss = F.cross_entropy(x, target)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # exponential moving average
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2
            torch.save(self.net.state_dict(),
                       self.args.dataset + '.pt')
            self.logger.info("Error (epoch %d): %.4f" % (epoch, loss_avg))

def test(net, test_loader):
    net.eval()
    loss_avg = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader: 
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)

            loss = F.cross_entropy(output, target)
        
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            loss_avg += float(loss.data)

    net.train()
    return correct / len(test_loader.dataset)

def test_cls(net, test_loader):
    net.eval()
    loss_avg = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)

            loss = F.cross_entropy(output, target)

            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            loss_avg += float(loss.data)

    print(loss_avg / len(test_loader))
    return correct / len(test_loader.dataset)


def get_acc(w):
    cumsum = np.insert(np.cumsum(w), 0, 0)
    reverse_cumsum = np.insert(np.cumsum(w[::-1]), 0, 0)[::-1]
    # Obtain accuracy vector
    acc = cumsum - reverse_cumsum
    return acc

def find_best_hyp(w, n):
    acc = get_acc(w)
    best_hyp = np.argmax(acc)
    return best_hyp

def version_space_reduction(graph, query_idxs, I, J, B1, B2, n, w, spend):

    # BASE CASE: return total amount spent
    if B2 == 0:
        return spend, query_idxs

    # RECURSIVE CASE: narrow down version space
    budget = min(B1, B2)
    # Construct lambda
    lmbda = []
    for _ in range(I):
        lmbda.append(0)
    for _ in range(I, J + 1):
        lmbda.append(1 / (J - I + 1))
    for _ in range(J + 1, n):
        lmbda.append(0)
    lmbda = np.array(lmbda)

    # Sample according to lambda without replacement.
    # Step 1: Create a new lambda vector which will reflect probabilities for only unlabeled indices.
    lmbda_unlabeled = []
    for i, node in enumerate(graph):
        if node.labeled:
            lmbda_unlabeled.append(0)
        else:
            lmbda_unlabeled.append(lmbda[i])

    # Step 2: Sample according to lmbda_unlabeled
    # Step 2.1: If every lambda is 0, terminate by returning spend
    if sum(lmbda_unlabeled) == 0:
        return spend, query_idxs
    # Step 2.2: Sample budget or if budget is too large, sample all nonzero indices
    greater_than_zero = [num for num in lmbda_unlabeled if num > 0]
    num_sample = min(budget, len(greater_than_zero))
    samp_idxs = np.random.choice(n, size=num_sample, replace=False,
                                    p=np.array(lmbda_unlabeled / sum(lmbda_unlabeled)))

    # Step 3: Update query_idxs and spend tracker
    for idx in samp_idxs:
        assert not graph[idx].labeled
        query_idxs.add(graph[idx].idx)
        spend += 1

    # Step 4: Label every samp_idx and update w
    for i in samp_idxs:
        graph[i].update()
        if graph[i].label == 1:
            w[i] = 1
        else:
            w[i] = -1

    # Step 5: Compute diameter of uncertainty region. We only count unlabeled examples
    cur_diam = 0
    for m in range(I, J + 1):
        if not graph[m].labeled:
            cur_diam += 1

    # Shrink version space
    if cur_diam > 0:
        # Step 2: Compute shrinkage factor
        base = B2 // B1
        if base <= 1:
            c2 = 1
        else:
            # c1 = math.log(cur_diam, B2 // B1 + 1)
            c2 = cur_diam ** (1 / (B2 // B1))

        # Step 3: Iteratively shrink version space
        diam = cur_diam

        while diam > math.ceil(cur_diam / c2):
            # Compare accuracy of left and right hypothesis
            accs = get_acc(w)
            # Step 3.3: Determine which accuracy is higher. Shift the pointer of the lower accuracy hypothesis
            if accs[I] < accs[J]:
                if not graph[I].labeled:
                    diam -= 1
                I += 1
            elif accs[I] > accs[J]:
                if not graph[J].labeled:
                    diam -= 1
                J -= 1
            else:
                b = np.random.randint(0, 2)
                if b == 0:
                    if not graph[I].labeled:
                        diam -= 1
                    I += 1
                else:
                    if not graph[J].labeled:
                        diam -= 1
                    J -= 1

    return version_space_reduction(graph, query_idxs, I, J, B1, B2 - budget, n, w, spend)


if __name__ == '__main__':
    '''
    define argument parser
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'PACS'])
    # dataset related
    parser.add_argument('--aux_out_dataset', type=str, default='iNaturalist',
                        choices=['svhn', 'lsun_c', 'lsun_r',
                            'isun', 'textures', 'places',
                            'tinyimages_300k', 'cifar100', 'iNaturalist'],
                        help='Auxiliary out of distribution dataset')
    parser.add_argument('--test_out_dataset', type=str,
                        choices=['svhn', 'lsun_c', 'lsun_r',
                                    'isun', 'textures', 'places', 'tinyimages_300k', 'cifar100', 'iNaturalist'],
                        default='iNaturalist', help='Test out of distribution dataset')

    parser.add_argument('--model', default='resnet50', type=str, help='model architecture: [resnet18, wrt40, wrt28, wrt16, densenet100, resnet50, resnet34]')
    parser.add_argument('--head', default='mlp', type=str, help='either mlp or linear head')
    parser.add_argument('--feat_dim', default = 128, type=int,
                    help='feature dim')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=4,
                        help='Pre-fetching threads.')
    # deep model params
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_class', type=int, default=10)

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--ood_rate', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--train_binary_classifier', type=int, default=0)
    parser.add_argument('--load_full_classifier', type=int, default=0)
    parser.add_argument('--ft_epochs', type=int, default=50) 
    parser.add_argument('--ft_weight', type=float, default=10)

    parser.add_argument('--loss_add', type=int, default=0)
    parser.add_argument('--load_filtered_data', type=int, default=0)
    parser.add_argument('--direct_test_ood', type=int, default=0)
    parser.add_argument('--num_sing_vectors', type=int, default=1)

    parser.add_argument('--ood_rate_1', type=float, default=0.5, help='proportion of covariate-shifted ood data in wild dataset')
    parser.add_argument('--ood_rate_2', type=float, default=0.1, help='proportion of semantic-shifted ood data in wild dataset')
    parser.add_argument('--cortype', type=str, default='gaussian_noise', help='corrupted type of images')

    parser.add_argument('--scores', type=str, default='energy', choices=['entropy', 'LCS', 'margin', 'energy'])
    parser.add_argument('--budget', type=float, default=1000, choices=[40, 100, 500, 1000, 2000, 5000, 10000])
    parser.add_argument('--strategy', type=str, default='aha', choices=['topk', 'random', 'aha'])
    parser.add_argument('--id_rate', type=float, default=0.1)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--mode', default='online', choices=['online', 'disabled'], help='whether disable wandb logging')
    parser.add_argument('--mu', default=7, type=int,
                                    help='coefficient of unlabeled batch size')

    parser.add_argument('--target_domain', type=str, default='photo', choices=['sketch', 'photo', 'art_painting', 'cartoon'])

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    name = '{}_{}'.format(args.strategy, args.aux_out_dataset)
    log_dir = 'log/{}'.format(args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = MyLog(os.path.join(log_dir, name + '.log'))
    logger.info(args)

    date_time = datetime.now().strftime("%d_%m_%H:%M")

    args.name = date_time + "_" + 'ood_{}_ft_weight_{}_idrate_{}_oodmix_{}_oodrate1_{}_oodrate2_{}_scores_{}_strategy_{}_budget_{}_cortype_{}_cortype_test_{}_lr_{}_batch_size_{}_epochs_{}'.format(args.aux_out_dataset, args.ft_weight, args.id_rate, args.ood_mix_rate, args.ood_rate_1, args.ood_rate_2, args.scores, args.strategy, args.budget, args.cortype, args.cortype_test, args.lr, args.batch_size, args.ft_epochs)

    seed = args.seed
    device = 'cuda:0'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed) 
    random.seed(seed)  

    state = {k: v for k, v in args._get_kwargs()}

    from make_dataset_aha import make_datasets_aha

    train_loader_in, train_loader_in_large_bs, train_loader_aux_in, \
    train_loader_aux_cor, train_loader_aux_out, test_loader_in, \
    test_loader_cor, test_loader_out = make_datasets_aha(
        args.dataset, args.aux_out_dataset, args.test_out_dataset, args.ood_rate_1, args.ood_rate_2, state, args.cortype, args.id_rate, args.target_domain)


    print("\n len(train_loader_in.dataset) {} " \
          "len(train_loader_aux_in.dataset) {}, " \
          "len(train_loader_aux_cor.dataset) {}, " \
          "len(train_loader_aux_out.dataset) {}, " \
          "len(test_loader_in.dataset) {}, " \
          "len(test_loader_cor.dataset) {}, " \
          "len(test_loader_ood.dataset) {}, ".format(
        len(train_loader_in.dataset),
        len(train_loader_aux_in.dataset),
        len(train_loader_aux_cor.dataset),
        len(train_loader_aux_out.dataset),
        len(test_loader_in.dataset),
        len(test_loader_cor.dataset),
        len(test_loader_out.dataset)))


    wandb.init(
        project='aha',
        entity='ood-project',
        name=args.name,
        mode=args.mode,
        config=args)


    if args.woods_setting:

        se = Sever(args,
                   args.optim,
                   logger,
                   eta=args.lr,
                   batch_size=args.batch_size,
                   num_rounds=args.num_rounds,
                   num_epochs=args.num_epochs,
                   seed=args.seed)
        se.rng = np.random.default_rng(args.seed)


        # LOAD PRETRAINED CLASSIFIER
        # for CIFAR-10
        if args.load_full_classifier:
            if args.num_class == 10:
                se.net.load_state_dict(torch.load('./checkpoints/cifar/cifar10_wrn_pretrained.pt'))
            else:
                se.net.load_state_dict(torch.load('./checkpoints/cifar/cifar100_wrn_pretrained.pt'))
                test_cls(se.net, train_loader_in)
        else:
            se.net.load_state_dict(torch.load(args.dataset + '.pt'))

        se.net.eval()

        index_in = 0
        batch_iterator = iter(train_loader_aux_in)

        if args.aux_out_dataset in ['textures', 'lsun_r', 'lsun_c']:
            batch_iterator_out = iter(train_loader_aux_out)
        else:
            batch_iterator_out = iter(train_loader_aux_cor)



        for data_all, target in train_loader_in:
            data = data_all.cuda()
            output_in = se.net(data)

            if args.scores == 'entropy':
                probs = F.softmax(output_in, dim=1)
                log_probs = torch.log(probs)
                scores = -(probs*log_probs).sum(1)

            elif args.scores == 'LCS':
                probs = F.softmax(output_in, dim=1)
                scores = -probs.max(1)[0]

            elif args.scores == 'margin':
                probs = F.softmax(output_in, dim=1)
                probs_sorted, idxs = probs.sort(descending=True)
                scores = probs_sorted[:,1] - probs_sorted[:, 0]

            elif args.scores == 'energy':
                scores = -torch.logsumexp(output_in, dim=1)

            elif args.scores == 'random':
                probs = F.softmax(output_in, dim=1)
                log_probs = torch.log(probs)
                scores = -(probs*log_probs).sum(1)

            if index_in == 0:
                scores_all_in = scores.detach().cpu()

            else:
                scores_all_in = np.concatenate([scores_all_in, scores.detach().cpu()], 0)

            index_in += 1
            if index_in % 10 == 0:
                print(index_in * 128)

        scores_all_in = np.array(scores_all_in)
        thred = np.quantile(scores_all_in, 0.95)
        index = 0

        if args.aux_out_dataset in ['textures', 'lsun_r', 'lsun_c']:
            train_loader_aux = iter(train_loader_aux_cor)
        else:
            train_loader_aux = iter(train_loader_aux_out)

        for data_all, target in train_loader_aux: 
            data = data_all

            try:
                in_set = next(batch_iterator)
                out_set = next(batch_iterator_out)
            except StopIteration:
                batch_iterator = iter(train_loader_aux_in)
                in_set = next(batch_iterator)

                if args.aux_out_dataset in ['textures', 'lsun_r', 'lsun_c']:
                    batch_iterator = iter(train_loader_aux_out)
                else:
                    batch_iterator = iter(train_loader_aux_cor)                

                out_set = next(batch_iterator)

            aux_set = torch.cat([data, in_set[0], out_set[0]], 0)

            if index == 0:
                keep_data_all = aux_set
            else:
                keep_data_all = torch.cat([keep_data_all, aux_set], 0)


            if args.aux_out_dataset in ['textures', 'lsun_r', 'lsun_c']:
                mask_label = np.concatenate([np.ones(len(data)),
                            np.zeros(len(in_set[0])), 2*np.ones(len(out_set[0]))], 0) 
                mask_class = np.concatenate([target,
                             in_set[1], -1*np.ones(len(out_set[0]))], 0)
                
            else:
                mask_label = np.concatenate([2*np.ones(len(data)),
                            np.zeros(len(in_set[0])), np.ones(len(out_set[0]))], 0)  
                mask_class = np.concatenate([-1*np.ones(len(target)),
                            in_set[1], out_set[1]], 0)
                
            labels_direct = np.concatenate([np.ones(len(data)),
                                np.zeros(len(in_set[0])), np.zeros(len(out_set[0]))], 0)

            aux_set = aux_set.cuda()
            output = se.net(aux_set)

            if args.scores == 'entropy':
                probs = F.softmax(output, dim=1)
                log_probs = torch.log(probs)
                scores = -(probs*log_probs).sum(1)

            elif args.scores == 'LCS':
                probs = F.softmax(output, dim=1)
                scores = -probs.max(1)[0]

            elif args.scores == 'margin':
                probs = F.softmax(output, dim=1)
                probs_sorted, idxs = probs.sort(descending=True)
                scores = probs_sorted[:,1] - probs_sorted[:, 0]

            elif args.scores == 'energy':
                scores = -torch.logsumexp(output, dim=1)

            elif args.scores == 'random':
                probs = F.softmax(output, dim=1)
                log_probs = torch.log(probs)
                scores = -(probs*log_probs).sum(1)

            if index == 0:
                scores_all = scores.detach().cpu()
                mask_label_all = mask_label
                target_all = mask_class
                data_set_all = aux_set.detach().cpu()
                direct_label_all = labels_direct
            else:
                scores_all = np.concatenate([scores_all, scores.detach().cpu()], 0)
                mask_label_all = np.concatenate([mask_label_all, mask_label], 0)
                target_all = np.concatenate([target_all, mask_class], 0)
                data_set_all = np.concatenate([data_set_all, aux_set.detach().cpu()], 0)
                direct_label_all = np.concatenate([direct_label_all, labels_direct], 0)

            index += 1
            if index % 10 == 0:
                print(index * 128)


        if args.strategy == 'topk':
            _, indices_select = torch.from_numpy(scores_all).topk(int(args.budget), dim=0, largest=True, sorted=True)
            selected = torch.from_numpy(scores_all[indices_select])
            data_set_selected = torch.from_numpy(data_set_all[indices_select])
            selected_label = torch.from_numpy(mask_label_all[indices_select])
            selected_category = torch.from_numpy(target_all[indices_select])

        elif args.strategy == 'random':
            indices_select = torch.from_numpy(np.random.permutation(len(scores_all))[:int(args.budget)])
            selected = torch.from_numpy(scores_all[indices_select])
            data_set_selected = torch.from_numpy(data_set_all[indices_select])
            selected_label = torch.from_numpy(mask_label_all[indices_select])
            selected_category = torch.from_numpy(target_all[indices_select])

        elif args.strategy == 'aha':
            nodes = []

            start_time = time.time()
            start_datetime = datetime.now()

            for idx, (margin, label) in enumerate(zip(scores_all, direct_label_all)):
                nodes.append(Node(idx, margin, label))

            # List of nodes separated by class where each list of nodes is sorted by scores
            sort_idx = np.argsort(-scores_all)
            graph = [nodes[idx] for idx in sort_idx]
            reverse_graph = [nodes[idx] for idx in sort_idx[::-1]]

            # This is what we return. This will be the additional indices that we query on this iteration.
            query_idxs = set()
            # This tracks how much we have spent on version space reduction.
            version_spend = 0
            # Number of parallel annotators.
            #B1 = self.strategy_config["B1"]
            B1 = 1

            B2 = args.budget // 2

            # VERSION SPACE REDUCTION
            # Start collect examples and label.

            # Initialize w
            w = []
            for node in graph:
                if not node.labeled:
                    w.append(0)
                elif node.label == 1:
                    w.append(1)
                else:
                    w.append(-1)
            # Find region of uncertainty and perform version reduction
            # Step 1: Find I*
            I = 0
            for idx, node in enumerate(graph):
                # I* must be labeled
                if not node.labeled:
                    continue
                # I* is found
                if node.label != 1:
                    break
                I = idx
            # Step 2: Find J*
            J = 0
            for idx, node in enumerate(reverse_graph):
                # J* must be labeled
                if not node.labeled:
                    continue
                # J* is found
                if node.label == 1:
                    break
                J = idx
            # Step 3: Update J to due to prior reversal of graph
            J = len(graph) - J - 1
            # Step 4: Update B2 if there are too few unlabeled examples
            unlabeled = 0
            for node in graph:
                if not node.labeled:
                    unlabeled += 1
            B2 = min(B2, unlabeled)
            # Step 5: Spend half of budget reducing the version space.
            spend = 0
            spend, query_idxs = version_space_reduction(graph, query_idxs, I, J, B1, B2, len(graph), np.array(w), spend)
            version_spend += spend
            
            #QUERY AROUND HYPOTHESIS
            rem_budget = args.budget - version_spend

            # Initialize w
            w = []
            for node in graph:
                if not node.labeled:
                    w.append(0)
                elif node.label == 1:
                    w.append(1)
                else:
                    w.append(-1)
            # Find best hypothesis
            best_hyp = find_best_hyp(w, len(graph))
            # Query around this hypothesis

            end_time = time.time()
            end_datetime = datetime.now()

            total_time = end_time - start_time

            print("best_hyp")
            print(best_hyp)

            wandb.log({'best_hyp': best_hyp})

            right = best_hyp
            left = right - 1
            bit = 1
            left_exceed, right_exceed = False, False
            # Step 2: Keep moving away from point and query labels
            class_budget = rem_budget
            #class_budget = args.budget

            while class_budget > 0:
                # Update bit
                bit = (bit + 1) % 2
                # We have labeled as much as we can; break
                if left_exceed and right_exceed:
                    break
                # Label to left
                if bit == 0:
                    if left < 0:
                        left_exceed = True
                        continue
                    if not graph[left].labeled:
                        query_idxs.add(graph[left].idx)
                        graph[left].update()
                        class_budget -= 1
                    left -= 1
                # Label to right
                else:
                    if right >= len(direct_label_all):
                        right_exceed = True
                        continue
                    if not graph[right].labeled:
                        query_idxs.add(graph[right].idx)
                        graph[right].update()
                        class_budget -= 1
                    right += 1
            selected = torch.from_numpy(scores_all[list(query_idxs)])
            data_set_selected = torch.from_numpy(data_set_all[list(query_idxs)])
            selected_label = torch.from_numpy(mask_label_all[list(query_idxs)])
            selected_category = torch.from_numpy(target_all[list(query_idxs)])


    # for CIFAR-10
    binary_classifier = WideResNet(40, args.num_class, 2, dropRate=0.3).cuda()

    logistic_regression = torch.nn.Sequential(
        torch.nn.Linear(128, 1))

    logistic_regression = logistic_regression.cuda().train()

    binary_classifier.train()
    binary_cls_optimizer = torch.optim.SGD(list(binary_classifier.parameters()) + list(logistic_regression.parameters()),
                                            momentum=0.9,
                                            nesterov=True,
                                        lr=0.001, weight_decay=args.weight_decay)

    if args.load_full_classifier:
        if args.num_class == 10:
            binary_classifier.load_state_dict(torch.load(
                './checkpoints/cifar/cifar10_wrn_pretrained.pt'))
        else:
            binary_classifier.load_state_dict(torch.load(
                './checkpoints/cifar/cifar100_wrn_pretrained.pt'))

    else:
        binary_classifier.load_state_dict(torch.load(args.dataset + '.pt'))


    batch_begin_in = 0
    batch_begin_ood = 0
    criterion = torch.nn.CrossEntropyLoss()

    binary_scheduler = torch.optim.lr_scheduler.LambdaLR(
        binary_cls_optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.ft_epochs * len(train_loader_in),
            1,  
            1e-6 / 0.001))

    index_selected_ood = np.argwhere(np.array(torch.squeeze(selected_label)==2)) # ood
    index_selected_in = np.argwhere(np.array(torch.squeeze(selected_label)==0)) # in
    index_selected_cor = np.argwhere(np.array(torch.squeeze(selected_label)==1)) # cor
    index_selected_id = np.argwhere(np.array(torch.squeeze(selected_label)!=2)) # in + cor

    index_selected_ood = np.squeeze(index_selected_ood)
    index_selected_id = np.squeeze(index_selected_id)

    data_set_selected = torch.squeeze(data_set_selected)

    selected_in_data = data_set_selected[torch.from_numpy(index_selected_id)].cuda()
    selected_ood_data = data_set_selected[torch.from_numpy(index_selected_ood)].cuda()

    selected_in_category = selected_category[torch.from_numpy(index_selected_id)].cuda()
    selected_in_category = torch.squeeze(selected_in_category)

    selected_indata_length = len(selected_in_data)
    selected_ooddata_length = len(selected_ood_data)

    print("selected_indata_length")
    print(selected_indata_length)
    print("selected_ooddata_length")
    print(selected_ooddata_length)

    permutation_idx_in = torch.randperm(selected_indata_length)
    permutation_idx_ood = torch.randperm(selected_ooddata_length)
     
    for epoch in range(args.ft_epochs):

        for in_set in train_loader_in:
            binary_cls_optimizer.zero_grad()

            if selected_indata_length - batch_begin_in < args.batch_size:
                batch_begin_in = 0
                permutation_idx_in = torch.randperm(selected_indata_length)
            if selected_ooddata_length - batch_begin_ood < args.batch_size:
                batch_begin_ood = 0
                permutation_idx_ood = torch.randperm(selected_ooddata_length)

            in_data_batch = selected_in_data[permutation_idx_in][batch_begin_in:batch_begin_in + args.batch_size].cuda()
            in_data_batch_size = len(in_data_batch)
            ood_data_batch = selected_ood_data[permutation_idx_ood][batch_begin_ood:batch_begin_ood + args.batch_size].cuda()
            ood_data_batch_size = len(ood_data_batch)

            out, out_logits = binary_classifier.forward_my1(torch.cat([in_set[0].cuda(), in_data_batch, ood_data_batch], 0))
            loss_s = F.cross_entropy(out_logits[:len(in_set[1]) + len(selected_in_category[batch_begin_in:batch_begin_in + args.batch_size])], torch.cat([in_set[1].cuda(), selected_in_category[permutation_idx_in][batch_begin_in:batch_begin_in + args.batch_size].type(torch.LongTensor).cuda()], 0))
            output1 = logistic_regression(out)

            binary_labels = torch.ones(len(in_set[1]) + ood_data_batch_size).cuda() # for CIFAR-10
            binary_labels[len(in_set[1]):] = 0
            output2 = torch.cat([output1[:len(in_set[1])], output1[len(in_set[1]) + in_data_batch_size:len(in_set[1]) + in_data_batch_size + ood_data_batch_size]], 0)

            energy_reg_loss = F.binary_cross_entropy_with_logits(output2.view(-1), binary_labels.float())
            loss = loss_s + args.ft_weight * energy_reg_loss

            batch_begin_in += args.batch_size
            batch_begin_ood += args.batch_size

            loss.backward()
            binary_cls_optimizer.step()
            binary_scheduler.step()

        acc = test(binary_classifier, test_loader_in)
        print('Epoch: ', epoch, 'Acc:', acc)
        print('Loss: ', loss)

        wandb.log({'loss_s': loss_s, 'energy_reg_loss': energy_reg_loss, 'loss': loss, 'acc': acc})

    binary_classifier.eval()
    logistic_regression.eval()

    wandb.log({'selected_id':len(index_selected_in), 'selected_cor':len(index_selected_cor), 'selected_ood':len(selected_ood_data)})

    def test_ood_function(test_loader_in, load_ood=False):
        index = 0
        with torch.no_grad():

            for in_set in test_loader_in:
                out, _ = binary_classifier.forward_my1(in_set[0].cuda())
                if index == 0:
                    score_all = logistic_regression(out).view(-1)
                else:
                    score_all = torch.cat([energy_all, logistic_regression(out).view(-1)], -1)
                index += 1
        return score_all.cpu().detach().numpy()

    score_id = test_ood_function(test_loader_in, load_ood=False)
    score_ood = test_ood_function(test_loader_out, load_ood=True)

    acc_id = test(binary_classifier, test_loader_in)
    acc_ood = test(binary_classifier, test_loader_cor)

    from utils.metric_utils import get_measures, print_measures

    measures = get_measures(score_id, score_ood, plot=False)
    print_measures(measures[0], measures[1], measures[2], 'energy')
    print('Acc ID: ', acc_id, 'Acc OOD: ', acc_ood)

    summary_metrics = {
            'FPR95': measures[2],
            'AUROC': measures[0],
            'AUPR': measures[1],
            'Accuracy ID': acc_id,
            'Accuracy OOD': acc_ood}

    for metric_name, metric_value in summary_metrics.items():
        wandb.summary[metric_name] = metric_value

    wandb.log({'FPR95': measures[2], 'AUROC': measures[0], 'AUPR': measures[1], 'Accuracy ID': acc_id, 'Accuracy OOD': acc_ood})
