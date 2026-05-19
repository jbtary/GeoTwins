# First import useful functions
# To read DAS data
import nc_array_processing as dnc
# General python libraries
import os
import sys
import numpy as np
from skimage.util import view_as_blocks, view_as_windows
import torch
import copy
from tqdm import tqdm

# Specific DASN2N functions
from dasn2n import DASN2N
# Build an absolute path from this notebook's parent directory
module_path = os.path.abspath('dasn2n')
# Add to sys.path if not already present
if module_path not in sys.path:
    sys.path.append(module_path)
# Now you can import the desired function or class
from model import ArrayDataset

# Define functions to train the u-net model
def load_data(fname,start_ch,end_ch):
    '''
    Loading the individual DAS h5 files and cutting a range of channels
    '''
    xsec_febus = dnc.read.febus(fname)
    xsec_febus = dnc.process.decimate(xsec_febus,target_rate=50,exact_rate=True)
    xsec_febus_rmed, median_febus = dnc.process.remove_median(xsec_febus)
    xsec_febus_rmed['data'] = xsec_febus_rmed['data'][start_ch-1:end_ch,:]
    xsec_febus_rmed['header']['channels'] = [start_ch-1,end_ch]
    return xsec_febus_rmed

def pre_proc_data(data):
    '''
    Preprocessing following publication of Lapins 2024
    Take DAS data after basic processing and convert it to data ready for torch training
    '''
    # If swapping axes is necessary (to have time x channels)
    das_numpy_array = np.swapaxes(data, 0, 1)
    # Remove mean and normalize
    offset = np.mean(das_numpy_array, axis=0, keepdims=True) # Mean along time axis (one per channel)
    das_numpy_array = das_numpy_array - offset
    norm_factor = np.std(das_numpy_array, axis=0, keepdims=True) # std along time axis (one per channel)
    das_numpy_array = das_numpy_array / norm_factor
    # Make sure data type is float32 (required for torch model):
    if das_numpy_array.dtype != 'float32':
        das_numpy_array = das_numpy_array.astype('float32')

    # Pad and split data into blocks for model processing (adding what's needed to have round number of blocks)
    das_array_pad = np.pad(das_numpy_array, 
        ((0, mod.INPUT_SHAPE[-2] - (das_numpy_array.shape[-2] % mod.INPUT_SHAPE[-2])), 
        (0, mod.INPUT_SHAPE[-1] - (das_numpy_array.shape[-1] % mod.INPUT_SHAPE[-1]))), 
        mode='reflect')

    # Window data (not overlapping)
    das_array_blocks = view_as_blocks(das_array_pad, tuple(mod.INPUT_SHAPE))

    # Clear some memory
    del(das_numpy_array)
    del(das_array_pad)

    # Reshape to 3D array to feed into model (no_blocks, samples, channels)
    model_in = np.reshape(das_array_blocks, (-1, das_array_blocks.shape[2], das_array_blocks.shape[3]))

    # Clear some memory
    del(das_array_blocks)
    return model_in

def get_data_loaders(model_in,batch_size=24):
    '''
    '''
    model_in_dataset = ArrayDataset(model_in) # Convert dataset from numpy to torch
    model_in_loader = torch.utils.data.DataLoader(model_in_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  pin_memory=True)
    return model_in_loader

def calc_loss(pred,targets):
    loss = torch.nn.functional.mse_loss(pred,targets)
    return loss

def train_model(unet,optimizer,scheduler,pathdata,night_files,day_files,num_epochs=30):
    '''
    To do: 
    - Do a test/validation part
    - Implement data augmentations (flipping, resampling data, cutmix/mixup, synthetic signals)
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model_wts = copy.deepcopy(unet.state_dict())
    best_loss = 1e10

    with open('training_log.txt','w') as f:
        f.write('Starting training.\n')
        f.write('Device used:\n')
        f.write(str(device)+'\n')

    for epoch in range(num_epochs):
        with open('training_log.txt','a') as f:
            f.write('Epoch {}/{}\n'.format(epoch+1, num_epochs))
            f.write('----------\n')
        
        for param_group in optimizer.param_groups:
            with open('training_log.txt','a') as f:
                f.write('LR '+str(param_group['lr'])+'\n')

        unet.train()  # Set model to training mode
        epoch_samples = 0
        cumloss = 0
        
        # Loop on the data files (tqdm(range(0,len(night_files))))
        for x in range(0,len(night_files)):
            curhr = int(night_files[x][21:23]) # Current night hour

            # First import the Night data (target)
            train_loader = {}
            data = load_data(pathdata+night_files[x],1,191) # Use only first part for now
            model_in = pre_proc_data(data['data'])
            train_loader['night'] = get_data_loaders(model_in)

            # Second import the Day data (to be cleaned)
            #data = load_data(pathdata+day_files[x],1,105)
            # Use day file 9 hours after night file (apart from 17/10 that stops at 9:16)
            if int(night_files[x][18:20]) < 17:
                data = load_data(pathdata+night_files[x][0:21]+
                                 '{:02d}'.format(curhr+9)+night_files[x][23::],1,191)
            else:
                data = load_data(pathdata+night_files[x][0:18]+
                                 '16_'+'{:02d}'.format(curhr+15)+night_files[x][23::],1,191)
            
            model_in = pre_proc_data(data['data'])
            train_loader['day'] = get_data_loaders(model_in)
            del(data)
            del(model_in)

            nightiter = iter(train_loader['night'])
            dayiter = iter(train_loader['day'])

            for y in range(0,len(train_loader['night'])):
                nightpatch = next(nightiter)
                daypatch = next(dayiter)
                
                # Determine if model is using half precision (should not be the case)
                first_param = next(mod.parameters(), None)
                if first_param is not None and first_param.dtype == torch.float16:
                    nightpatch = nightpatch.half()
                    daypatch = daypatch.half()
                if device != 'cpu':
                    nightpatch = nightpatch.to(device)
                    daypatch = daypatch.to(device)
            
                optimizer.zero_grad() # zero the parameter gradients
                outputs = unet(daypatch)
                loss = calc_loss(outputs,nightpatch)
                cumloss += loss.data.cpu().numpy()

                loss.backward()
                optimizer.step()
                epoch_samples += nightpatch.size(0) # Batch amount

        scheduler.step()
        epoch_loss = cumloss / epoch_samples
        with open('training_log.txt','a') as f:
            f.write('Epoch loss is '+str(epoch_loss)+' (Total samples: '+str(epoch_samples)+', cumulative loss: '+str(cumloss)+')\n')

        # deep copy the model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(unet.state_dict())
    
    with open('training_log.txt','a') as f:
        f.write('Best val loss: {:4f}\n'.format(best_loss))

    # load best model weights
    unet.load_state_dict(best_model_wts)
    return unet,best_model_wts

def run(mod,modelname):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = mod.to(device)
    optimizer_ft = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer_ft, milestones=[15, 25], gamma=0.1)
    #lr_sched = torch.optim.lr_scheduler.StepLR(optimizer_ft,step_size=20,gamma=0.1)
    #lr_sched = torch.optim.lr_scheduler.ConstantLR(optimizer_ft,factor=1,total_iters=30)

    model_fin,wghts = train_model(model, optimizer_ft, lr_sched, pathdata, night_files, day_files, num_epochs=30)
    torch.save(model_fin, modelname)
    torch.cuda.empty_cache()
    return model_fin,wghts

# Get the original dasn2n architecture
mod = DASN2N()
# If you want to change the input shape of the model:
#mod.INPUT_SHAPE = [512,96] # time samples x number of channels (original is 128 x 96)

# Preparation of data to train the u-net model
pathdata = '/mnt/geofib/1/2025-GRANGEGORMAN_NOISE/2025-10-07_Grangegorman/'
febus_files = []
for x in os.listdir(pathdata):
    if x.endswith(".h5"):
        febus_files.append(x)

febus_files.sort()

# Make list of day and night files (day: noisy data / night: target data)
day_files = []
night_files = []
for x in febus_files:
    if int(x[21:23]) >= 5:
        day_files.append(x)
    else:
        if int(x[21:23]) == 0 and int(x[24:26]) < 30:
            day_files.append(x)
        else:
            night_files.append(x)


model_fin,wghts = run(mod,"trained_dasn2n.pt")
