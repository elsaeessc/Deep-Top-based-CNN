# Convolutions: 1 --> 128 --> 64 MaxPool 64 --> 64 MaxPool
# After Flatten: 64 --> 256 --> 256 --> 2
# ReLu, ausser letzte mit Softmax
# Adadelta mit lr = 0,003, Scheduler mit Threshold 0.005 min fuer Val_Loss, Faktor = 1/(2**(1/2)) Batchsize = 128

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pandas

import time

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# This time, we have converted the information of the energy of the jet constituents into an image. Each image has 40x40 pixels = 1600 pixels. Each column represents the *colour* intensity in each pixel.
# 
# Let's define a function, `to_image`, that rewrites these columns as a 40x40 numpy matrix, with one additional index, that represents the colours of the image (see later boxes).

# In[20]:


# 1 image has 40x40 pixels = 1600 pixels

print("Begin Data Preparing")
pixels = ["img_{0}".format(i) for i in range(1600)]

def to_image(df):
    return  np.expand_dims(np.expand_dims(df[pixels], axis=-1).reshape(-1,40,40), axis=1)

# Read the train sample
store_train_images = pandas.HDFStore("train_img.h5")
df_train_images = store_train_images.select("table",stop=1200000)

images_train = to_image(df_train_images)
images_train_labels = df_train_images["is_signal_new"].to_numpy()

# Read the validation sample
store_val_images = pandas.HDFStore("val_img.h5")
df_val_images = store_val_images.select("table",stop= 400000)
    
images_val = to_image(df_val_images)
images_val_labels = df_val_images["is_signal_new"].to_numpy()

print("Data Prepararing Finished")

# Convolutional Architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.padd = nn.ZeroPad2d((1, 2, 1, 2))

        self.conv1 = nn.Conv2d(1, 128, 4, padding = False)
        self.conv2 = nn.Conv2d(128, 64, 4, padding = False)
        self.conv3 = nn.Conv2d(64, 64, 4, padding = False)
        
        self.pool = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(10 * 10 * 64, 64) 
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 2)
        
    def forward(self, x):
        
        x = F.relu(self.conv1(self.padd(x)))
        x = F.relu(self.conv2(self.padd(x)))
        x = self.pool(x)
        
        x = F.relu(self.conv3(self.padd(x)))
        x = F.relu(self.conv3(self.padd(x)))
        x = self.pool(x)
        
        x = x.view(-1, 10 * 10 * 64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = nn.Softmax(dim=1)(self.fc4(x))
        
        return x
  

# Which device to use for NN calculations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

# Loss function
criterion = nn.CrossEntropyLoss()

# how many epochs to train for
n_epochs = 100

# how many examples / batch
batch_size = 128

# Keep track of the accuracies, losses and AUC
train_accs = []
val_accs = []
train_losses = []
val_losses = []
train_rocs = []
val_rocs = []
train_AUCs = []
val_AUCs = []

print("Start Training")
model = CNN().to(device)
X_train = images_train
y_train = images_train_labels
y_train_tensor = torch.tensor(y_train,dtype=torch.long).to(device)
    
X_val = images_val
y_val = images_val_labels
y_val_tensor = torch.tensor(y_val,dtype=torch.long).to(device)

#Adadelta optimizer with learning rate 0.3
optimizer = torch.optim.Adadelta(model.parameters(), lr=0.03)

#scheduler used to optimze the optimizer
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=1/np.sqrt(2), threshold=0.0005)

def Predictor_Zwolf(arr):
    y_prediction = np.empty([np.shape(arr)[0],2])
    for i in range(1200):
        i_start = i*1000
        i_stop = (i+1)*1000
        X_batch = torch.tensor(arr[i_start:i_stop], dtype=torch.float)
        y_prediction_batch = model(X_batch.to(device)).detach().cpu().numpy()
        for j in range(1000):
            y_prediction[i*1000+j] = y_prediction_batch[j]
    return y_prediction

def Predictor_Vier(arr):
    y_prediction = np.empty([np.shape(arr)[0],2])
    for i in range(400):
        i_start = i*1000
        i_stop = (i+1)*1000
        X_batch = torch.tensor(arr[i_start:i_stop], dtype=torch.float)
        y_prediction_batch = model(X_batch.to(device)).detach().cpu().numpy()
        for j in range(1000):
            y_prediction[i*1000+j] = y_prediction_batch[j]
    return y_prediction

# Create network object
train_examples = X_train.shape[0]
n_batches = int(train_examples/batch_size)

# Loop over the epochs
for ep in range(n_epochs):
    
    ep_start = time.time()
    # reorder the training events for each epoch
    idx = np.arange(X_train.shape[0])
    np.random.shuffle(idx)
    
    X_train = X_train[idx]
    y_train = y_train[idx]
    y_train_tensor = y_train_tensor[idx]
    
    loop_start = time.time()
    # Each epoch is a complete loop over the training data
    for i in range(n_batches):
        
        # Reset gradient
        optimizer.zero_grad()
        
        i_start = i*batch_size
        i_stop  = (i+1)*batch_size
        
        # Convert x and y to proper objects for PyTorch
        x = torch.tensor(X_train[i_start:i_stop],dtype=torch.float).to(device)
        y = torch.tensor(y_train[i_start:i_stop],dtype=torch.long).to(device)

        # Apply the network 
        net_out = model(x)
        
        # Calculate the loss function
        loss = criterion(net_out,y)
                
        # Calculate the gradients
        loss.backward()
        
        # Update the weights
        optimizer.step()
    # end of loop over batches
    loop_end = time.time()
        
    # Calculate predictions on training and validation data
    pred_start = time.time()
    y_pred_train = Predictor_Zwolf(X_train)
    y_pred_val = Predictor_Vier(X_val)
    pred_end = time.time()
    
    y_pred_train_tensor = torch.tensor(y_pred_train,dtype=torch.float).to(device)
    y_pred_val_tensor = torch.tensor(y_pred_val,dtype=torch.float).to(device)

    # Calculate accuracy on training and validation data
    train_acc = sum(y_train == np.argmax(y_pred_train,axis=1)) / y_train.shape[0]
    val_acc = sum(y_val == np.argmax(y_pred_val,axis=1)) / y_val.shape[0]
    
    # Calculate loss of train and val data
    train_loss = criterion(y_pred_train_tensor,y_train_tensor)
    val_loss = criterion(y_pred_val_tensor,y_val_tensor)
    scheduler.step(val_loss)
    
    # Calculate ROC
    train_roc = roc_curve(y_train, y_pred_train[0:1200000,1])
    val_roc = roc_curve(y_val, y_pred_val[0:400000,1])
    
    # Calculate auc
    train_auc = roc_auc_score(y_train, y_pred_train[0:1200000,1])
    val_auc = roc_auc_score(y_val, y_pred_val[0:400000,1])
    
    # and store the values for later use
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_rocs.append(train_roc)
    val_rocs.append(val_roc)
    train_AUCs.append(train_auc)
    val_AUCs.append(val_auc)
    
    ep_end = time.time()
    
    if ep%5 == 0:
        np.save("train_acc", np.array(train_accs))
        np.save("val_acc", np.array(val_accs))
        np.save("val_auc", np.array(val_AUCs))
        np.save("train_auc", np.array(train_AUCs))
        np.save("train_roc", np.array(train_rocs))
        np.save("val_roc", np.array(val_rocs))
        np.save("train_loss", np.array(train_losses))
        np.save("val_loss", np.array(val_losses))
    
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f"model_epoche_{ep}.pt")
    print(f"epoch: {ep}, Train Acc: {train_acc}, Val. Acc: {val_acc} Val AUC: {val_auc}, Ep time {ep_end - ep_start}, Loop time {loop_end - loop_start}, Pred time {pred_end - pred_start}")
# end of loop over epochs

np.save("train_acc", np.array(train_accs))
np.save("val_acc", np.array(val_accs))
np.save("val_auc", np.array(val_AUCs))
np.save("train_auc", np.array(train_AUCs))
np.save("train_roc", np.array(train_rocs))
np.save("val_roc", np.array(val_rocs))
np.save("train_loss", np.array(train_losses))
np.save("val_loss", np.array(val_losses))
    
print("Finish")