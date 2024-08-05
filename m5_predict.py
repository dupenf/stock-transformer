# Reference: https://github.com/ctxj/Time-Series-Transformer-Pytorch/tree/main
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import copy
import math
import time
import matplotlib.pyplot as plt


from m3_train import predict
from d2_datasets import get_batch, get_data
from d1_features import log_features



model = torch.load("saved_weights.pt")
log_prices = log_features()
train_data, test_data = get_data(log_prices, 0.9)
predicted_seq, real_seq = predict(model, test_data)

fig2, ax2 = plt.subplots(1, 1)

ax2.plot(predicted_seq, color='red', alpha=0.7)
ax2.plot(real_seq, color='blue', linewidth=0.7)
ax2.legend(['Actual', 'Forecast'])
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Log Prices')

fig2.tight_layout()

