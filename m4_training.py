# Reference: https://github.com/ctxj/Time-Series-Transformer-Pytorch/tree/main
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import copy
import math
import time
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from m1_transformer import Transformer
from m3_train import evaluate


from a0_config import device, output_window, input_window, batch_size, USE_CUDA
from d2_datasets import get_batch, get_data
from d1_features import log_features

################################################################################
model = Transformer().to(device)
criterion = nn.MSELoss()
lr = 0.00005
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
################################################################################

log_prices = log_features()
train_data, test_data = get_data(log_prices, 0.9)
################################################################################

N_EPOCHS = 150
for epoch in range(1, N_EPOCHS + 1):
    epoch_start_time = time.time()
    model.train()  # Turn on the evaluation mode
    total_loss = 0.0
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss = total_loss + loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches | "
                "lr {:02.10f} | {:5.2f} ms | "
                "loss {:5.7f}".format(
                    epoch,
                    batch,
                    len(train_data) // batch_size,
                    scheduler.get_last_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss,
                )
            )
            total_loss = 0
            start_time = time.time()

    if epoch % N_EPOCHS == 0:  # Valid model after last training epoch
        val_loss = evaluate(model, test_data, criterion=criterion)
        print("-" * 80)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | valid loss: {:5.7f}".format(
                epoch, (time.time() - epoch_start_time), val_loss
            )
        )
        print("-" * 80)

    else:
        print("-" * 80)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s".format(
                epoch, (time.time() - epoch_start_time)
            )
        )
        print("-" * 80)

    scheduler.step()

torch.save(model, "saved_weights.pt")
