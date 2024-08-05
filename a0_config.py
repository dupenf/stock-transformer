import torch


USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
input_window = 7 # number of input time steps
output_window = 1 # number of prediction steps (equals to one)
batch_size = 100