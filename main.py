import argparse
import time
import datetime
import pandas as pd
import math
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.optim import Adam, ASGD, SGD
import data
from model import RNNModel
import json


# def run():
parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='humber of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=20,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', type=bool, default=False,
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model_Adam_.pt',
                    help='path to save the final model')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', type=bool, default=True,
                    help='tie the word embedding and softmax weights')
parser.add_argument('--brown', action='store_true',
                    help='use the brown corpus from nltk')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

# print("args.data", args.data)
# print("args.model", args.model)
# print("args.emsize", args.emsize)
# print("args.nhid", args.nhid)
# print("args.nlayers", args.nlayers)
# print("args.lr", args.lr)
# print("args.clip", args.clip)
# print("args.epochs", args.epochs)
# print("args.batch_size", args.batch_size)
# print("args.bptt", args.bptt)
# print("args.seed", args.seed)
# print("args.cuda", args.cuda)
# print("args.log_interval", args.log_interval)
# print("args.save", args.save)
# print("args.dropout", args.dropout)
# print("args.tied", args.tied)

corpus = data.Corpus(args.data, args.brown)

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

eval_batch_size = 10
print("Batchifying data...")
train_data = batchify(corpus.train, args.batch_size)
print(train_data.size())
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
print("Instantiating Model...")
model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################


# def clip_gradient(model, clip):
#     """Computes a gradient clipping coefficient based on gradient norm."""
#     totalnorm = 0
#     for p in model.parameters():
#         modulenorm = p.grad.data.norm()
#         totalnorm += modulenorm ** 2
#     totalnorm = math.sqrt(totalnorm)
#     return min(1, args.clip / (totalnorm + 1e-6))


def clip_grad_norm(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.
    """
    parameters = list(parameters)
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef >= 1:
        return
    for p in parameters:
        p.grad.data.mul_(clip_coef)


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def evaluate(data_source):
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)

train_results = {"type": [], "epoch": [], "batch": [], "lr": [],
                 "time": [], "loss": [], "ppl": []}


def train():
    total_loss = 0
    start_time = time.time()
    cur_loss = np.Inf
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    optimizer = SGD(model.parameters(), lr=lr)
    for batch, i in enumerate(range(0, len(train_data) - 1, args.bptt)):
        # zero grad
        optimizer.zero_grad()
        # model.zero_grad()
        # get data
        data, targets = get_batch(train_data, i)
        hidden = repackage_hidden(hidden)
        # get outputs from models
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        if cur_loss == np.Inf:
            cur_loss = loss.data[0]
        else:
            cur_loss = 0.9 * cur_loss + 0.1 * loss.data[0]
        clip_grad_norm(model.parameters(), args.clip)

        # clipped_lr = lr * clip_gradient(model, args.clip)
        # for p in model.parameters():
        #     p.data.add_(-clipped_lr, p.grad.data)

        # take step
        optimizer.step()

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} |'
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            train_results["epoch"].append(epoch)
            train_results["batch"].append(batch)
            train_results["lr"].append(lr)
            train_results["time"].append(elapsed * 1000 / args.log_interval)
            train_results["loss"].append(cur_loss)
            train_results["ppl"].append(math.exp(cur_loss))
            train_results["type"].append("train")
            total_loss = 0
            start_time = time.time()
    opt_name = str(optimizer.__class__).split(".")[2]
    return opt_name

# Loop over epochs.
lr = args.lr
prev_val_loss = None
for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    opt_name = train()
    val_loss = evaluate(val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, math.exp(val_loss)))
    print('-' * 89)
    train_results["epoch"].append(epoch)
    train_results["batch"].append("Nan")
    train_results["lr"].append("Nan")
    train_results["time"].append(time.time() - epoch_start_time)
    train_results["loss"].append(val_loss)
    train_results["ppl"].append(math.exp(val_loss))
    train_results["type"].append("val")
    # Anneal the learning rate.
    if prev_val_loss and val_loss > prev_val_loss:
        lr /= 3.0
    prev_val_loss = val_loss

train_ = pd.DataFrame(train_results)

# Run on test data and save the model.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
with open("word_emb_Adam.json", 'wb') as outfile:
    json.dump(model.encoder.weight.data.cpu().numpy().tolist(), outfile)
train_results["epoch"].append("Nan")
train_results["batch"].append("Nan")
train_results["lr"].append("Nan")
train_results["time"].append("Nan")
train_results["loss"].append(test_loss)
train_results["ppl"].append(math.exp(test_loss))
train_results["type"].append("test")
df = pd.DataFrame(train_results)
today = "_".join(str(datetime.date.today()).split("-"))
file_name = opt_name + "_drp" + str(args.dropout) + "_lyr" + str(args.nlayers) + "_" + args.model + "_" + str(args.emsize) + "_" + str(args.nhid) + "_" + str(args.nlayers) + "_" + today + "penn_results.csv"
df.to_csv(file_name)
if args.save != '':
    with open(args.save, 'wb') as f:
        torch.save(model, f)
