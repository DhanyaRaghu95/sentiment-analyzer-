"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import pickle
import random


inputs = pickle.load(open("vecForSen.p","r"))
targets = pickle.load(open("ratings.p","r"))
testRand = []
print len(inputs)
for i in xrange(100):
    testRand.append(random.randrange(1000))
testInput = []
testTargets = []
print len(testRand)
for i in testRand:
    #print i
    testInput.append(inputs[i])
    testTargets.append(targets[i])
testRand.sort(reverse = True)
for i in testRand:
    inputs.pop(i)
    targets.pop(i)
    #inputs.pop(i)
#print(type(inputs))
#exit(0)

# data I/O
#data = open('input.txt', 'r').read() # should be simple plain text file
#chars = list(set(data))
#data_size, vocab_size = len(data), len(chars)
#print 'data has %d characters, %d unique.' % (data_size, vocab_size)	#32, 14
#char_to_ix = { ch:i for i,ch in enumerate(chars) }
#ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 4 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1
vocab_size = 32

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(1, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((1, 1)) # output bias
#print("by",by)

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  nxs, hs, ys, ps,xs = {}, {}, {}, {},{}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  l = len(inputs)-1
  for t in xrange(len(inputs)):
    nxs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    nxs[t] = inputs[t]
    xs[t] = [[i] for i in nxs[t]]
    xs[t] = np.array(xs[t])
    #print xs[t],"%%%"

    #print np.array(inputs[t]).shape

    #print "!!!!!!!!!!!!!!!!!!!!!",xs[t]
    #

    #print "$$$$$$$$$$$$$$$$$$$$$$$$$$$",xs[t]
    #xs[t][inputs[t]] = 1	#one hot vector
    #print(xs[t],"iiiiiiii")	# len = 14
    #print Wxh.shape, xs[t].shape
    #exit(0)
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    #print(hs[t],"hhhhhhhh", len(hs[t]))	# a vector of h, one for each hidden layer circle, all in one square.
    #print(Why,"Why",len(Why))
    temp = np.dot(Why, hs[t])
    #print(temp,"temp",len(temp))
  ys[l] = np.dot(Why, hs[l]) + by # unnormalized log probabilities for next char
    # ys is a vector.
  ps[l] = np.exp(ys[l]) / np.sum(np.exp(ys[l])) # probabilities for next chars
  ps[l] = ys[l]
  #print("pssss",ps)
  #print(len(ps[t]),"#########################")	# size of one output unit is 14
  #print("ys",ys[t],len(ys[t]))
    #print("###################33333",[targets[t],0])
  #print targets
  #print "aaaaaaaa",ps,ps[l][0],targets
  loss += ps[l][0] - targets #-np.log(ps[l][targets,0]) # softmax (cross-entropy loss)
  #print loss
  #exit(0)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])

  ######################################################################
  #backpropagate
  #print dy.shape
  dy = np.copy(ps[l])
  #print dy.shape
  dy[0] -= 1 # backprop into y
  dWhy += np.dot(dy, hs[l].T)
  #print dby.shape,dy.shape

  dby += dy
  dh = np.dot(Why.T, dy) + dhnext
  for t in reversed(xrange(len(inputs)-1)):

    '''print("*********************")
    print(dy)
    print("!!!!!!!!!!!!!")
    print(dby)'''

    #dh = dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
    dh = dhnext
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1], ps[l][0]

def sample(h, seed_ix, n):
  """
  sample a sequence of integers from the model
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

acc = 0
tp = 0
fn = 0
fp = 0
n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0


while n<len(inputs):
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  #if p+seq_length+1 >= len(data) or n == 0:
  hprev = np.zeros((hidden_size,1)) # reset RNN memory
  p = 0 # go from start of data
  # inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  # targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
  # inputs=[]
  # targets=[2,3,1,4,1]
  # print("input",len(inputs))
  # print("targets",targets)

  '''# sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )
    '''
  # forward seq_length characters through the net and fetch gradient

  if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
  epoch = 0
  while epoch<10:
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev, ps = lossFun(inputs[n], targets[n], hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    #print n,"LOSS: ",loss,"  smooth : ",smooth_loss,type(smooth_loss)
    #exit(0)


    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
      mem += dparam * dparam
      param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
    epoch += 1
  #print "aaaaaaaa",ps,targets[n]
  if (ps<1 and targets[n]==0):
    acc+=1
    tp+=1
  if (ps>=1 and targets[n]==1):
    acc+=1
    fp+=1
  else:
    fn+=1
  p += seq_length # move data pointer
  n += 1 # iteration counter


for i in xrange(len(testInput)):
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev, ps = lossFun(inputs[i], targets[i], hprev)
  if (ps<1 and targets[i]==0):
    acc+=1
    tp+=1
  if (ps>=1 and targets[i]==1):
    acc+=1
    fp+=1
  else:
    fn+=1
print acc