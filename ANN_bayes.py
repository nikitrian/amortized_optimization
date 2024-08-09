# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import pickle # Quick saving format
import tqdm
import math


torch.manual_seed(10)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# %%
with open('device.txt', 'w') as f:
    f.write(device)
    f.write(f'\n{torch.cuda.is_available()}')


def save_pkl(data, save_name):
    """
    Saves .pkl file from of data in folder: tmp/
    """
    with open(save_name, 'wb') as handle:
        pickle.dump(data, handle, protocol = pickle.HIGHEST_PROTOCOL)
    #print(f'File saved at: {save_name}')
    return None

def load_pkl(file_name):
    """
    Loads .pkl file from path: file_name
    """
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)
    print(f'Data loaded: {file_name}')
    return data

# %%
def ANN_BAYES(p, space, train_set, test_set, normP):
    
    p = unnormalize_x(p, space)

    epochs = int(np.round(p[0].item()))
    lr = math.exp(p[1])
    HLsize = int(np.round(p[2].item()))
    num = int(np.round(p[3].item()))


    print(f"Epochs: {epochs} | Learning rate: {lr:.8f} | HL size: {HLsize} | number of HL: {num} ")
    no_inputs = train_set[0].shape[1]
    no_outputs = train_set[1].shape[1]
    m = ReLUNet(no_inputs, HLsize, no_outputs, num).to(device).to(torch.float64)
    m, (duration, lossplot, accplot), acc = train_model(m, lr, epochs, train_set, test_set)

    # Extract hyperparameters
    hyperparams = f"epochs_{epochs}_layers_{num}_neurons_{HLsize}_lr_{lr:.8f}"

    # Save loss plot as SVG with dpi=600 and tight layout
    plt.plot(lossplot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Plot - {hyperparams}')
    plt.tight_layout()
    plt.savefig(f'loss_ACC_{acc*100:.2f}_{hyperparams}.jpeg', format='jpeg', dpi=600)
    plt.close()

    # Save accuracy plot as SVG with dpi=600 and tight layout
    plt.plot(accplot)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Plot - {hyperparams}')
    plt.tight_layout()
    plt.savefig(f'accuracy_ACC_{acc*100:.2f}_{hyperparams}.jpeg', format='jpeg', dpi=600)
    plt.close()

    save_model(m, normP, f'ann_ACC_{acc*100:.2f}_{epochs}_{lr:.4f}_{HLsize}_{num}.pkl', epochs, lr, HLsize, num)   
    #print('Accuracy:',acc.item().detach().to('cpu')*100,'%')
    
    return acc

# Sample level accuracy for total correctness
def sample_level_accuracy(predictions, targets):
    predictions = predictions.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()

    accuracy = sum(((targets == predictions).sum(axis = 1) - 6).clip(0))/targets.shape[0]
    return accuracy

# Feed-forward neural network architecture
class ReLUNet(torch.nn.Module):
    """
    ReLU neural network structure.
    x hidden layers, 1 output layer.
    size of input, output, and hidden layers are specified
    """
    def __init__(self, n_input, n_hidden, n_output, num_layers):
        super().__init__()

        self.hidden_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.hidden_layers.append(torch.nn.Linear(n_input, n_hidden))
            else:
                self.hidden_layers.append(torch.nn.Linear(n_hidden, n_hidden))


        self.output = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.output(x)
        return x


def train_model(m, lr, epochs, train_set, test_set):
    """
    Train the ANN
    """
    import time
    start = time.time()


    # Setup data to be device agnostic
    device = next(m.parameters()).device
    X = torch.tensor(train_set[0]).to(device)
    Y = torch.tensor(train_set[1]).to(device)
    Xt = torch.tensor(test_set[0]).to(device)
    Yt = torch.tensor(test_set[1]).to(device)
    Y= Y.squeeze()
    Yt= Yt.squeeze()

    
    optimizer = torch.optim.Adam(m.parameters(), lr = lr)
    loss_func = torch.nn.BCEWithLogitsLoss()


    lossplot = []
    accplot = []
    
    for epc in range(epochs):

        optimizer.zero_grad()

        y_logits = m(X)

        y_pred = torch.round(torch.sigmoid(y_logits))

    
        loss = loss_func(y_logits, Y)
        
        loss.backward()
        optimizer.step()

        ## Testing
        m.eval()
        with torch.no_grad():
            # 1. Forward pass
            test_logits = m(Xt)
            test_pred = torch.round(torch.sigmoid(test_logits))
        
            # 2. Calculate loss and acc
            test_loss = loss_func(test_logits, Yt)

            test_acc = sample_level_accuracy(test_pred, Yt)

        # Print out what's happening
        #if epc % 100 == 0:
            #print(f"Epoch: {epc} | Loss: {loss:.15f} Acc: {acc:.15f} | Test loss: {test_loss:.15f} Test acc: {test_acc:.15f}")

        lossplot.append(loss.cpu().item())
        accplot.append(test_acc)
    end = time.time()
    duration = end - start
    print(f'Time taken: {duration//60}m {duration%60:.2f}s')
    return m, (duration, lossplot, accplot), test_acc

def mm_norm(arr, normP):
    """
    Min max normalisation
    """
    arrmax = normP[0] 
    arrmin = normP[1]
    return (arr - arrmin)/(arrmax - arrmin)

def mm_rev(norm, normP):
    """
    Reverse min max normalisation
    """
    arrmax = normP[0] 
    arrmin = normP[1]
    return norm*(arrmax - arrmin) + arrmin

def save_model(m, normP, mname, epochs, lr, HLsize, num):
    """
    Save model parameters
    """
    import copy
    cpu_m = copy.copy(m)
    cpu_m.to('cpu')
    labels = list(cpu_m.state_dict().keys())
    SD = cpu_m.state_dict()
    structure = [epochs, lr, HLsize, num]
    P = {}
    P['structure'] = structure
    P['state_dict'] = SD
    P['normP'] = normP

    save_pkl(P, mname)
    return None


# %%
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from gpytorch.kernels import MaternKernel
from botorch.models.transforms import Standardize



def model_initialization(X, y, GP=None, state_dict=None, *GP_args, **GP_kwargs):
    """
    Fit GP model. The function accepts
    state_dict is used to initialize the GP model.
    
    Parameters
    ----------
    X : Input data
        
    y : Output data
        
    GP : GPyTorch model class
        
    state_dict : GP model state dict
        
    Returns
    -------
    mll : Marginal loglikelihood
    
    gp : Updated GP
    """

    covar_module = MaternKernel(nu=2.5)


    if GP is None:
        GP = SingleTaskGP
        
    model = GP(X, y,  outcome_transform=Standardize(1), covar_module = covar_module, *GP_args, **GP_kwargs).to(X)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model

# %%

from botorch.optim import optimize_acqf
def bo_step(X, y, objective, bounds, GP=None, acquisition=None, q=1, state_dict=None):
    """
    Bayesian optimization!
    Each iteration includes:
        1. Fit GP model using (X, y)
        2. Define acquisition function
        3. Optimize acquisition function to obtain candidate point
        4. Evaluate objective function at candidate point
        5. Add new point to the data set

    """

    # Create GP model
    mll, gp = model_initialization(X, y, GP=GP, state_dict=state_dict)
    fit_gpytorch_model(mll)
    
    # Create acquisition function
    acquisition = acquisition(gp)
    
    # Optimize acquisition function
    candidate  = optimize_acqf(
        acquisition, bounds=bounds, q=q, num_restarts=50, raw_samples=1024,
    )
    
    
    X = torch.cat([X, candidate[0]])
    y = torch.cat([y, objective(candidate[0].squeeze())])


    return X, y, gp


# %%

# %%
from botorch.utils.transforms import normalize, unnormalize

def normalize_x(params, space):
    bounds = torch.tensor([var['domain'] for var in space]).to(params).t()
    params = normalize(params, bounds)
    return params

def unnormalize_x(params, space):
    bounds = torch.tensor([var['domain'] for var in space]).to(params).t()
    params = unnormalize(params, bounds)
    return params

def convert_tensor_to_dict_list(X, space):
    """
    Convert a tensor to a list of dictionaries.
    This function is adapted from [yeahrmek's GitHub Repository]
    """
    def _wrap_row(row):
        wrapped_row = {}
        for i, x in enumerate(row):
            wrapped_row[space[i]['name']] = x.item()
        
            if space[i]['type'] == 'discrete':
                wrapped_row[space[i]['name']] = int(np.round(x.item()))
        return wrapped_row
    
    wrapped_X = []
    for i in range(X.shape[0]):
        wrapped_X.append(_wrap_row(X[i]))
        
    return wrapped_X


# %%
# LOAD data
data = pd.read_csv('mip_demands.csv')

X = data.drop(["profile","m11", "m22","m33","m44","m55","m66","inff"], axis=1)
y = data[["m11", "m22","m33","m44","m55","m66","inf"]]

# Split data into train and test sets
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    random_state=42) # make the random split reproducible


Xtrain = X_train.to_numpy(dtype = 'float64')
Ytrain = y_train.to_numpy(dtype = 'float64')
Xtest = X_test.to_numpy(dtype = 'float64')
Ytest = y_test.to_numpy(dtype = 'float64')

# Min max normalisation
Xmax = Xtrain.max(axis = 0)
Xmin = Xtrain.min(axis = 0)
XnormP = (Xmax, Xmin)

normP = XnormP

train_set = (mm_norm(Xtrain, normP), Ytrain)
test_set = (mm_norm(Xtest, normP), Ytest)

# %%
# %%
from botorch.utils.sampling import draw_sobol_samples


# Define the hyperparameter search region
space = [
    {'name': 'epochs', 'type': 'discrete', 'domain': (500, 15000)},
    {'name': 'learning rate', 'type': 'continuous', 'domain': (math.log(1e-5), math.log(1e-1))},
    {'name': 'hidden layer size', 'type': 'discrete', 'domain': (50, 256)},
    {'name': 'hidden layer number', 'type': 'discrete', 'domain': (1, 3)}
]

bounds_01 = torch.zeros(2, len(space), dtype=torch.float64)
bounds_01[1] = 1

# Draw 5 samples through Sobol sequence to initialize the GP
init_X = draw_sobol_samples(bounds_01, 5, 1).squeeze()

init_y =[]


for i in range(len(init_X)):
    init_y.append(ANN_BAYES(init_X[i], space, train_set, test_set, normP))


init_y2= torch.tensor(init_y).reshape(-1,1)




# %%
from botorch.acquisition import (ExpectedImprovement, UpperConfidenceBound)

params = init_X
params = params.detach().to('cpu')
scores = init_y2
scores = scores.detach().to('cpu')


state_dict = None

iterations = 300

objective = lambda x: torch.tensor(ANN_BAYES(x, space, train_set, test_set, normP)).reshape(-1,1)



with tqdm.tqdm(total=iterations) as bar:
    while len(scores) < iterations:
        
        n_samples = len(scores)

        GP = SingleTaskGP

        #acquisition = lambda gp: ExpectedImprovement(gp, scores.max(), maximize=True)
        acquisition = lambda gp: UpperConfidenceBound(gp, beta=150, maximize=True)

        
        params, scores, gp = bo_step(params, scores, objective, bounds_01,
                                     GP=GP, acquisition=acquisition, 
                                     state_dict=state_dict)


        # Move the data back to CPU

        state_dict = gp.state_dict()
        
     
        bar.update(len(scores) - n_samples)

# %%
#=============================================================RESULTS==============================================================

def best_hyperparams(params, scores, space):
    parameters = unnormalize_x(params, space)
    parameters= convert_tensor_to_dict_list(parameters, space)

    best_idx = np.argmax(scores.cpu().numpy())
    print(best_idx)
    return parameters[best_idx]

best_param = best_hyperparams(params, scores, space)

print(f"Parameters: {best_param}")


# Print the maximum accuracy
print('MAX accuracy', scores.max().item())

# Move scores back to CPU for plotting
scores = scores.cpu().numpy()



# %%
cum_best = np.maximum.accumulate(scores)


# save cum_best as pickle
with open('cum_best.pkl', 'wb') as f:
    pickle.dump(cum_best, f)



# %%

# Plot the Bayesian optimization progress
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})  

iters=list(range(iterations))
# plot lines

plt.plot(iters, cum_best*1E2,'-o', label = "UCB", color= 'mediumvioletred')
plt.ylabel('Sample Level Accuracy (%)', fontsize=18)
plt.xlabel('# Iterations',  fontsize=18)
plt.legend()

plt.savefig('ann_bayes.svg',  bbox_inches='tight')



# %%
