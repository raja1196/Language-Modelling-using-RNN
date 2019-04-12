#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: count.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
from src.lstm import LSTMcell
import src.assign as assign
import matplotlib.pyplot as plt

def count_0_in_seq(input_seq, count_type):
    """ count number of digit '0' in input_seq

    Args:
        input_seq (list): input sequence encoded as one hot
            vectors with shape [num_digits, 10].
        count_type (str): type of task for counting. 
            'task1': Count number of all the '0' in the sequence.
            'task2': Count number of '0' after the first '2' in the sequence.
            'task3': Count number of '0' after '2' but erase by '3'.

    Return:
        counts (int)
    """

    if count_type == 'task1':
        # Count number of all the '0' in the sequence.
        # create LSTM cell
        cell = LSTMcell(in_dim=10, out_dim=1)
        # assign parameters
        assign.assign_weight_count_all_0_case_1(cell, in_dim=10, out_dim=1)
        # initial the first state
        prev_state = [0.]
        # read input sequence one by one to count the digits
        for idx, d in enumerate(input_seq):
            prev_state = cell.run_step([d], prev_state=prev_state)
        count_num = int(np.squeeze(prev_state))
        return count_num

    if count_type == 'task2':
        # Count number of '0' after the first '2' in the sequence.
        # create LSTM cell
        cell = LSTMcell(in_dim=10, out_dim=2)
        # assign parameters
        assign.assign_weight_count_all_0_after_2(cell, in_dim=10, out_dim=2)
        # initial the first state
        prev_state = [0.,0.]

        # read input sequence one by one to count the digits
        for idx, d in enumerate(input_seq):
            prev_state = cell.run_step([d], prev_state=prev_state)

        count_num = int(np.squeeze(prev_state[0][0]))

        return count_num

    if count_type == 'task3':
        # Count number of '0' in the sequence when receive '2', but erase
        # the counting when receive '3', and continue to count '0' from 0
        # until receive another '2'.
        # Count number of '0' after the first '2' in the sequence.
        # create LSTM cell
        cell = LSTMcell(in_dim=10, out_dim=2)
        # assign parameters

        assign.assign_weight_count_all_0_after_2_del_3(cell, in_dim=10, out_dim=2)
        # initial the first state
        prev_state = [0.,0.]

        # read input sequence one by one to count the digits
        for idx, d in enumerate(input_seq):
            #print("PREV_STATE = ", prev_state)
            prev_state = cell.run_step([d], prev_state=prev_state)

        #Code used to generate subplots for Proj 4 Part 1 Report
        '''
        plt.figure(2)
        plt.subplot(211)
        plt.title('Internal State Value for 0 Count')
        plt.plot(cell.gArrC0)
        plt.ylabel('c(t)[0]')
        plt.xlabel('t')
        plt.xticks(np.arange(0, len(cell.gArrC0), step=1))
        plt.subplot(212)
        plt.title('Internal State Value for 2 Count')
        plt.plot(cell.gArrC2)
        plt.ylabel('c(t)[1]')
        plt.xlabel('t')
        plt.xticks(np.arange(0, len(cell.gArrC0), step=1))
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                            wspace=0.35)
        plt.show()
        '''

        count_num = int(np.squeeze(prev_state[0][0]))

        return count_num



        