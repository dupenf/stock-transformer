# Reference: https://github.com/ctxj/Time-Series-Transformer-Pytorch/tree/main
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import copy
import math
import time
import matplotlib.pyplot as plt

from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer




def log_features():

    df = pd.read_csv("./datasets/sh.600000.csv")
    close = df['close']
    log_prices = np.diff(np.log(close))
    log_prices_csum = log_prices.cumsum() # Cumulative sum of log prices
    print(log_prices_csum)
    print("------------")
    print(log_prices)
    
    return log_prices


    # draw 
    fig1, ax1 = plt.subplots(2, 1)
    ax1[0].plot(close, color='red')
    ax1[0].set_title('Closed Price')
    ax1[0].set_xlabel('Time Steps')

    ax1[1].plot(log_prices_csum, color='blue')
    ax1[1].set_title('CSUM of Log Price')
    ax1[1].set_xlabel('Time Steps')

    fig1.tight_layout()
    
    
log_features()