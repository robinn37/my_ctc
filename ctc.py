#/usr/bin/env python

import math
import numpy as np
import time
import torch
import torch.nn as nn

NEG_INF = -float("inf")

def logsumexp(*args):
    if all(a==NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp


def log_softmax(acts, axis):
    acts = acts - np.max(acts, axis=axis, keepdims=True)
    probs = np.sum(np.exp(acts), axis=axis, keepdims=True)
    log_probs = acts - np.log(probs)
    return log_probs


def forward(log_probs, elabels, blank, enable_bypass=True):
    T, V = log_probs.shape
    L = elabels.shape[0]
    #print("T: {}, V:{}, L:{}".format(T, V, L))
    alphas = np.full((T, L), NEG_INF)
    alphas[0, 0] = log_probs[0, blank]
    alphas[0, 1] = log_probs[0, elabels[1]]
    bypass = 0
    for t in range(1, T):
        for l in range(L):
            if enable_bypass and l < (L-2) - 2*(T-1-t):
                bypass += 1
                continue # alphas[t, l] = NEG_INF
            else:
                alphas[t, l] = alphas[t-1, l]
                if l >= 1:  # blank or duplicate
                    alphas[t, l] = logsumexp(alphas[t, l], alphas[t-1, l-1])
                if l >= 2 and elabels[l] != elabels[l-2]: # different label
                    alphas[t, l] = logsumexp(alphas[t, l], alphas[t-1, l-2])
                alphas[t, l] = alphas[t, l] + log_probs[t, elabels[l]]
    #print('forward bypass rate: {}'.format(float(bypass)/(L*(T-1))))
    return alphas, -logsumexp(alphas[T-1, L-1], alphas[T-1, L-2])


def backward(log_probs, elabels, blank, enable_bypass=True):
    T, V = log_probs.shape
    L = elabels.shape[0]
    #print("T: {}, V:{}, L:{}".format(T, V, L))
    betas = np.full((T, L), NEG_INF)
    betas[T-1, L-2] = log_probs[T-1, elabels[L-2]]
    betas[T-1, L-1] = log_probs[T-1, blank] # elabels[L-1]
    bypass = 0
    for t in reversed(range(T-1)):
        for l in reversed(range(L)):
            if enable_bypass and l > 2*(t+1) + 1:
                bypass += 1
                continue # betas[t, l] = NEG_INF
            else:
                betas[t, l] = betas[t+1, l]
                if l < L-1:  # blank or duplicate
                    betas[t, l] = logsumexp(betas[t, l], betas[t+1, l+1])
                if l < L-2 and elabels[l] != elabels[l+2]: # different label
                    betas[t, l] = logsumexp(betas[t, l], betas[t+1, l+2])
                betas[t, l] = betas[t, l] + log_probs[t, elabels[l]]
    #print('backward bypass rate: {}'.format(float(bypass)/(L*(T-2))))
    return betas, -logsumexp(betas[0, 0], betas[0, 1])


def find_label_index(elabels, k):
    B = []
    for l, v in enumerate(elabels):
        if k == v:
            B.append(l)
    return B


'''
    Gradient computation on softmax output.
'''
def compute_gradient(log_probs, alphas, betas, elabels, blank):
    T, V = log_probs.shape
    L = elabels.shape[0]
    gradients = np.full(log_probs.shape, NEG_INF)
    log_p = logsumexp(betas[0, 0], betas[0, 1])
    B = []
    for t in range(T):
        for k in range(V):
            B = find_label_index(elabels, k)
            albetas = []
            for l in B:
                albetas.append(alphas[t, l] + betas[t, l])
            res = logsumexp(*albetas)
            gradients[t, k] = res - log_p - 2*log_probs[t, k]

    gradients = np.exp(gradients)
    gradients = -gradients
    return gradients


def test_logsumexp():
    inputs = np.random.rand(10)
    r = logsumexp(*inputs)
    cinputs = torch.from_numpy(inputs)
    torch_r = torch.logsumexp(cinputs, dim=0).detach().item()
    print(r)
    print(torch_r)
    assert np.allclose(r, torch_r, atol=1e-18, rtol=1e-18), \
        "logsumexp impl mismatch!" 


def test_log_softmax():
    inputs = np.random.rand(2, 10)
    log_probs1 = log_softmax(inputs, 1)
    print(log_probs1)
    print(np.sum(np.exp(log_probs1), axis=1, keepdims=True))
    cinputs = torch.from_numpy(inputs)
    log_probs2 = torch.nn.functional.log_softmax(cinputs, dim=1)
    assert np.allclose(log_probs1, log_probs2, atol=1e-8, rtol=1e-8), \
        "log_softmax impl mismatch!" 



def test(T, V, L):
    inputs = np.random.rand(T, V)
    labels = np.random.randint(1, V, L)
    #print('softmax input: {}'.format(inputs))
    #print('output labels: {}'.format(labels))
    log_probs = log_softmax(inputs, axis=1)
    #print('log_probs: {}'.format(log_probs))
    blank = 0
    elabels = [0]
    for l in labels:
        elabels.append(l)
        elabels.append(blank)
    elabels = np.array(elabels)
    start = time.time()
    alphas, loss = forward(log_probs, elabels, blank)
    print('loss: {}, used time: {}'.format(loss, time.time() - start))
    start = time.time()
    betas, loss_b = backward(log_probs, elabels, blank, enable_bypass=False)
    print('loss(backward): {}, used time: {}'.format(loss_b, time.time() - start))
    assert np.allclose(loss, loss_b, atol=1e-8, rtol=1e-8), \
        "forward and backward loss impl mismatch!" 

    # check loss by prob sum
    for _ in range(10):
        t = np.random.randint(1, T)
        log_prob_sum = NEG_INF
        for l in range(2*L+1):
            log_prob_sum = logsumexp(log_prob_sum,
                    alphas[t, l] + betas[t, l] - log_probs[t, elabels[l]])
        assert np.allclose(loss, -log_prob_sum, atol=1e-8, rtol=1e-8), \
            "forward and backward loss impl mismatch!" 


    # compare loss against pytorch ctc loss
    ctc_loss = nn.CTCLoss(reduction='none')
    ptinput = torch.from_numpy(log_probs)
    ptinput = ptinput.unsqueeze(0)
    ptinput = ptinput.transpose(0,1)
    target = torch.from_numpy(labels)
    input_lengths = torch.full(size=(1,), fill_value=T, dtype=torch.long)
    target_lengths = torch.full(size=(1,), fill_value=L, dtype=torch.long)
    start = time.time()
    pt_loss = ctc_loss(ptinput, target, input_lengths, target_lengths).item()
    print('pt_loss: {}, used time: {}'.format(pt_loss, time.time() - start))
    assert np.allclose(loss, pt_loss, atol=1e-8, rtol=1e-8), \
            "ctc loss is mismatched against pytorch impl"

    # check gradients
    grads = compute_gradient(log_probs, alphas, betas, elabels, blank)


if __name__ == '__main__':
    #test_logsumexp()
    #test_log_softmax()
    T = 200 # frames
    V = 50  # vocab size
    L = 20  # labels

    test(T, V, L)
