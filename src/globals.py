import torch

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu") # debugging index errors is easier on the cpu