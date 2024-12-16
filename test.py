import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import time
import torch.optim as optim
from model import SEST
import matplotlib.pyplot as plt
from data import DatasetFromHdf5
from torch.autograd import Variable
import scipy.io as sio
from tqdm import tqdm
from skimage import io


model_folder = "./checkpoints/"
save_path = "./srimg/ "

SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.benchmark = True
cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

test_set = DatasetFromHdf5(' ') # creat data for test
testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)

model = SEST()

path_checkpoint = model_folder + "{}.pth".format(350)
checkpoint = torch.load(path_checkpoint)
model.load_state_dict(checkpoint['net'])
model = model.cuda()

output_HRHSI_tol = np.zeros((11, 512, 512, 31))
GT_tol = np.zeros((11, 512, 512, 31))

print('Start Testing')
for iteration, batch in enumerate(tqdm(testing_data_loader), 1):
    GT, LRHSI, HRMSI = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
    LRHSI = LRHSI.cuda()
    HRMSI = HRMSI.cuda()

    with torch.no_grad():
        output_HRHSI, UP_LRHSI, Highpass = model(LRHSI, HRMSI)
    output_HRHSI_tol[iteration-1, :, :, :] = output_HRHSI.permute(0, 2, 3, 1).cpu().detach().numpy()
    GT_tol[iteration-1, :, :, :] = GT.permute(0, 2, 3, 1).cpu().numpy()

# save result
sio.savemat(save_path + 'SRHSI_data.mat', {'output': output_HRHSI_tol})
sio.savemat(save_path + 'GT.mat', {'GT': GT_tol})
print('Done')
