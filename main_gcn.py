     #!/usr/bin/python
import torch
from torch import nn
import optuna 

import neptune
import neptunecontrib.monitoring.optuna as opt_utils

import numpy as np 
import joblib

from model import Net
from data import load
import gcn_config as cfg
from Clustergcn import train
from functions import callback, EarlyStopping

from os.path import join
import time

import gc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def trainer(trial):
    since = time.time()
    i = mode_counter
    num_epochs = cfg.num_epochs
    
    best_loss = 100
    patience_count = 0

    ###Parameters###
    config = {'adj'     : cfg.adj_setup[i],
              'hidden'  : trial.suggest_int('hidden', 20, 1400, step=100),
              'lr'      : trial.suggest_loguniform('lr',1e-4, 1e-2),
              'dropout' : trial.suggest_discrete_uniform('dropout', 0.4, 0.8, 0.05)         }
    
    config['adj'][4] = trial.suggest_discrete_uniform('threshold',0.65,1,0.025)

    ###Model###                         
    in_features = np.load(join("FVS", "test_" + config['adj'][0] + ".npy")).shape[1]   
    model = Net(in_features, config['hidden'], 15, config['dropout']).to(device)
    
    ###Optimiser and Loss###
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()  
    
    ###Data###
    train_loader, train_size = load(config['adj'],"train")
    val_loader, val_size = load(config['adj'],"val")
    data = [{"train": train_loader, "val": val_loader},
            {"t_size": train_size, "v_size": val_size}]    
    
    ###Train & Test###
    for epoch in range(num_epochs):
        v_acc, v_loss  = train(model, data, optimizer, criterion)

        if v_loss < best_loss:
            best_loss = v_loss
            patience_count = 0
        if v_loss > best_loss:
            patience_count += 1
        if patience_count == cfg.patience:
            break
        trial.report(v_loss, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        gc.collect()   
    gc.collect() 
    
    trial_time = time.time() - since
    print("Trial Time: ",'%.1f' % trial_time, "Epoch:",epoch)  
    
    return v_loss

def main():
    neptune.init('manosathya/4YP')
    neptune.create_experiment(name='Loss Optimisation, final config:' + str(mode_counter))
    monitor = opt_utils.NeptuneMonitor()
    
    study = optuna.create_study(sampler = optuna.samplers.TPESampler(),
                                pruner = optuna.pruners.HyperbandPruner(),
                                direction='minimize')
    try:
        study.optimize(trainer, n_trials=cfg.num_trials, callbacks= [callback, monitor])
        opt_utils.log_study(study)
    except EarlyStopping:
        print("\n\nNo improvements in " + str(cfg.study_patience) + " trials")
    if cfg.save_study.upper() == "Y":    
        joblib.dump(study, join("Results", cfg.adj_setup[mode_counter][0] + "_final.pkl"))  
    trial = study.best_trial
    print("\nNumber of finished trials: ", len(study.trials))
    print("\nBest trial: " + str(trial.number) )
    print("  Loss: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))      
    
    
if __name__ == '__main__':
    global mode_counter
    for mode_counter in range(2,3):
        main()

    

