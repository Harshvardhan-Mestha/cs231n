from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = 10
   

    for i in range(0,num_train):
      scores = np.dot(X[i],W)
      scores -= np.max(scores)
      exp_scores = np.exp(scores)
      
      for j in range(0,num_classes):
        if j == y[i]:
          loss += -np.log(exp_scores[j]/np.sum(exp_scores))
          dW[:,j] = dW[:,j] + ((exp_scores[j]/np.sum(exp_scores))-1)*X[i]
        else:
          dW[:,j] = dW[:,j] + (exp_scores[j]/np.sum(exp_scores))*X[i]

      

    loss = loss/num_train
    loss += reg * np.sum(W * W)

    dW = dW/num_train
    dW += 2*reg*W
      

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = np.dot(X,W)
    exp_scores = np.exp(scores)
    #print(exp_scores.shape)
    correct = np.choose(y,exp_scores.T)
    sums = np.sum(exp_scores,axis=1)
    loss = np.sum(-np.log(correct/sums))

    norm_exp = (exp_scores.T/sums)
    norm_exp = norm_exp.T
    norm_exp[range(0,X.shape[0]),y] = (correct/sums)-1
    #print(norm_exp.shape)
    dW = np.dot(norm_exp.T,X)
    dW = dW.T

    #print(dW.shape)


    loss = loss/X.shape[0]
    loss = loss + 2*reg*np.sum(W*W)

    dW = dW/X.shape[0]
    dW = dW + 2*reg*W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
