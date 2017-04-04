import argparse
import time
import datetime
import pandas as pd
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import data
import model

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=50,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=50,
                    help='humber of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=6,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=20,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

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
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
print("Instantiating Model...")
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def clip_gradient(model, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, args.clip / (totalnorm + 1e-6))


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
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    optimizer = Adam(model.parameters(), lr=lr)
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

        clipped_lr = lr * clip_gradient(model, args.clip)
        for p in model.parameters():
            p.data.add_(-clipped_lr, p.grad.data)

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

# Loop over epochs.
lr = args.lr
prev_val_loss = None
for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    train()
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
        lr /= 4.0
    prev_val_loss = val_loss

train_ = pd.DataFrame(train_results)

# Run on test data and save the model.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
if args.save != '':
    with open(args.save, 'wb') as f:
        json.dump(data, outfile)
train_results["epoch"].append("Nan")
train_results["batch"].append("Nan")
train_results["lr"].append("Nan")
train_results["time"].append("Nan")
train_results["loss"].append(test_loss)
train_results["ppl"].append(math.exp(test_loss))
train_results["type"].append("test")
 
today = "_".join(str(datetime.date.today()).split("-"))
df = pd.DataFrame(train_results)
file_name = "Adam_drp0.65_" + args.model + "_" + str(args.emsize) + "_" + str(args.nhid) + "_" + str(args.nlayers) + "_" + today + "_results.csv"
df.to_csv(file_name)
# if args.save != '':
#     with open(args.save, 'wb') as f:
#         torch.save(model, f)
