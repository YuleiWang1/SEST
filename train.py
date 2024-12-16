import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import DatasetFromHdf5
from model import SEST
import numpy as np
import time
import matplotlib.pyplot as plt
from loss import *
from tqdm import tqdm

# ================== Pre-Define =================== #
SEED = 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.benchmark = True
cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# ============= HYPER PARAMS(Pre-Defined) ==========#
print('test')
lr = 1e-4
epochs = 350
ckpt_step = 50
batch_size = 16
model_folder = ""

model =SEST().to(device)

PLoss = HybridLoss(spectral_tv=True, spatial_tv=True).to(device)
# SLoss = spectral_loss().to(device)
# HLoss = HybridLoss().to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)   # optimizer 1
# optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)  # optimizer 2
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=70, gamma=0.5)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.1)

def save_checkpoint(model, ckpt_name):  # save model function

    model_out_path = model_folder + "{}.pth".format(ckpt_name)

    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
        "lr": lr
    }
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
    torch.save(checkpoint, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

train_set = DatasetFromHdf5('')  # creat data for training
training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
validate_set = DatasetFromHdf5('')  # creat data for validation
# put validate data to DataLoader for batches
validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=3, shuffle=True, pin_memory=True, drop_last=True)


print('Start training...')
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))

start_epoch = 0
for epoch in range(start_epoch, epochs, 1):

    epoch += 1
    epoch_train_loss, epoch_val_loss = [], []

    # ============Epoch Train=============== #
    model.train()

    for iteration, batch in enumerate(tqdm(training_data_loader), 1):
        GT, LRHSI, HRMSI = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        optimizer.zero_grad()  # fixed
        output_HRHSI, UP_LRHSI, Highpass = model(LRHSI, HRMSI)
        Pixelwise_Loss = PLoss(output_HRHSI, GT)

        Myloss = Pixelwise_Loss
        epoch_train_loss.append(Myloss.item())  # save all losses into a vector for one epoch

        Myloss.backward()  # fixed
        optimizer.step()  # fixed

        if iteration % 20 == 0:
            print("===>{} Epoch[{}]({}/{}): Loss: {:.6f}".format(time.ctime(), epoch, iteration, len(training_data_loader),
                                                               Myloss.item()))

    print("learning rate:º%f" % (optimizer.param_groups[0]['lr']))
    lr_scheduler.step()  # update lr

    t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
    print('Epoch: {}/{} training loss: {:.7f}'.format(epoch, epochs, t_loss))  # print loss for each epoch


    if epoch % ckpt_step == 0:  # if each ckpt epochs, then start to save model
        save_checkpoint(model, epoch)

    # ============Epoch Validate=============== #
    if epoch % 200 == 0:
        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(validate_data_loader, 1):
                GT, LRHSI, HRMSI = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                output_HRHSI, UP_LRHSI, Highpass = model(LRHSI, HRMSI)
                # output_HRHSI = model(LRHSI)
                # time_e = time.time()
                Pixelwise_Loss = PLoss(output_HRHSI, GT)
                MyVloss = Pixelwise_Loss
                epoch_val_loss.append(MyVloss.item())
        plt.figure(1)
        plt.ion()
        fig, axes = plt.subplots(ncols=2, nrows=2)
        LRHSI1 = LRHSI[0, [10, 20, 30], ...].float().permute(1, 2, 0).cpu().numpy()
        axes[0, 0].imshow(LRHSI1)
        axes[0, 1].imshow(HRMSI[0, ...].permute(1, 2, 0).cpu().numpy())
        axes[1, 0].imshow(output_HRHSI[0, [10, 20, 30], ...].permute(1, 2, 0).cpu().detach().numpy())
        axes[1, 1].imshow(GT[0, [10, 20, 30], ...].permute(1, 2, 0).cpu().numpy())
        plt.pause(0.1)
        v_loss = np.nanmean(np.array(epoch_val_loss))
        print("learning rate:º%f" % (optimizer.param_groups[0]['lr']))
        print('validate loss: {:.7f}'.format(v_loss))

