# Project 4 - Recurrent Neural Network (RNN)

- The goal of this project is for you to become familiar with a wildly used RNN unit, Long short-term memory (LSTM)
- This project contains two parts: 1. Design LSTM cells for counting digits in a sequence; 2. Language modeling using RNN with LSTM units.

# Requirements
- Python3
- Numpy
- One of the deep learning frameworks (only required for part 2)

# Dataset
- Part one does not require any dataset.
- Part two: Penn Tree Bank, can be downloaded from [Tomas Mikolov's webpage](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz).

# Part One Description 
## 1. Objective
- For this part, you need to manually choose the parameters for a LSTM cell to count the digit `0` in a sequence under different criteria. So you do not need to train the model.
- Given a digit sequence, choose three different sets of parameters for a LSTM cell for the following three tasks, respectively:
  1. Count number of all the digit `0` in the sequence;
  2. Count number of `0` after receiving the first `2` in the sequence;
  3. Count number of `0` after receiving the `2` in the sequence, but erase the counts after receiving `3`, then continue to count from 0 after receiving another `2`.
- For example, given the sequence `[1, 1, 0, 4, 3, 4, 0, 2, 0, 2, 0, 4, 3, 0, 2, 4, 5, 0, 9, 0, 4]`, the output for task 1 - 3 should be `7`, `5` and `2`.


## 3. Explaination for task 1.
- There are several suitable parameter sets for the task 1. Here we provide two examples, which can be found in [this script](part1/src/assign.py) (`assign_weight_count_all_0_case_1` and `assign_weight_count_all_0_case_2`). Both cases use 1-dimension internal state. 
- For case 1, the input gate, forget gate and output gate are always on (as value `1`). The input node gets value `1` when receives `0`, and value `0` when receives other digits. Thus, the internal state will be accumulated by 1 every time the input is digit `1`, so as the output.
- For case 2, the input node always gets value `1`, but the input gate only on when receives digit `0` and other two gates are always on. Thus, the output will be the same as case 1.
- To make things easy, I just use large values (`100` or `-100`) to make the sigmoid and hyperbolic functions saturating to get the state and the output value to be 1 or 0, so that I do not need to use any nonlinear activation function for the output of the LSTM cell.

# Part Two Description 
## 1. Objective
- Design a RNN with LSTM units for language modeling.
- Implement the RNN use one of deep learning frameworks.

## 2. Train the model
- Download PTB dataset from [Tomas Mikolov's webpage](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)
- Train the model using `simple-examples/data/ptb.char.train.txt` as training set. 
- Train the model using `simple-examples/data/ptb.train.txt` as training set.
- The text are already tokenized, so you can skip this step if you use this dataset.
- Try to generate text using both trained model.

## Authors

* **Rajaram Sivaramakrishnan** - [raja1196](https://github.com/raja1196)


## Reference
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [A Critical Review of Recurrent Neural Networks for Sequence Learning](https://arxiv.org/abs/1506.00019)
- [Visualizing and Understanding Recurrent Networks](https://arxiv.org/abs/1506.02078)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Language modeling a billion words](http://torch.ch/blog/2016/07/25/nce.html)
- [Tensorflow Language Modeling Tutorial](https://www.tensorflow.org/tutorials/sequences/recurrent)

