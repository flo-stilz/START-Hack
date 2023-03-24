# Training script
import numpy as np
import sys
import argparse
from datetime import datetime
from pathlib import Path
import ast
import re

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import os
sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from dataset.Dataset import *
from models.regressor_models import Regressor_Simple
import optuna
from optuna.integration import PyTorchLightningPruningCallback


def init_parser():
    # Settings
    parser = argparse.ArgumentParser(
        description='Hyperparameters for Prediction Task',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', type=int, required=False, default=8,
                        help='Batch size!')
    parser.add_argument('--lr', type=float, required=False, default=1e-4,
                        help='Learning Rate!')
    parser.add_argument('--drop', type=float, required=False, default=0.1,
                        help='Dropout of model')
    parser.add_argument('--reg', type=float, required=False, default=0,
                        help='Weight decay')
    parser.add_argument('--gamma', type=float, required=False, default=1.00,
                        help='Gamma value for Exponential LR-Decay')
    parser.add_argument('--acc_grad', type=int, required=False, default=1,
                        help='Set to 1 if no gradient accumulation is desired else enter desired value for accumulated batches before gradient step')
    parser.add_argument('--epochs', type=int, required=False, default=100,
                        help='Set maxs number of epochs.')
    parser.add_argument('--gpu', type=int, required=False, default=0,
                        help='Set the name of the GPU in the system')
    parser.add_argument('--model', type=str, required=False, default="Regressor_Simple",
                        help='Model architecture')
    parser.add_argument('--bert_model', type=str, required=False, default="distilbert-base-uncased",
                        help='BERT Model architecture')
    parser.add_argument('--rel_pred', action='store_true',
                        help='Add reliability prediction head')
    parser.add_argument('--num_layers', type=int, required=False, default=1,
                        help='Number of Linear Layers in regressor')
    parser.add_argument('--hidden_size', type=int, required=False, default=50,
                        help='Number of hidden units per layer in regressor')
    parser.add_argument('--hp_tuning', action='store_true',
                        help='Perform Random Search to tune hps')
    parser.add_argument('--trials', type=int, required=False, default=200,
                        help='Number of trials for hp tuning')
    parser.add_argument('--show_scores', type=bool, required=False, default=True,
                        help='Log ci and r2m scores')
    parser.add_argument('--no_ckpt', action='store_true',
                        help='Do not save model ckpt')
    parser.add_argument('--use_linear_skip', action='store_true',
                        help='Use Skip Connection Layers')
    parser.add_argument('--adj_lr', action='store_true',
                        help='Adjust lr by dividing it by 10 after val-loss plateaued')
    parser.add_argument('--num_models', type=int, required=False, default=2,
                        help='Amount of models used in ensemble method')
    parser.add_argument('--freeze', type=int, required=False, default=0,
                        help='Set how many layers to freeze')
    parser.add_argument('--freeze_emb', action='store_true',
                        help='Freeze embedding layer')
    parser.add_argument('--all-emb', action='store_true',
                        help='Use all embeddings not just CLS')
    parser.add_argument('--seq_length', type=int, required=True,
                        help='Amount of chat messages included')
    parser.add_argument('--num_labels', type=int, required=False, default=1,
                        help='How many values to predict for regression')

    opt = parser.parse_args()

    return opt


def train_model(opt, train_data, val_data, test_data):

    #train = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    val = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=8)
    test = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=8)

    # Initialize model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

    model = globals()[opt.model](opt=opt, train_set=train_data, val_set=val_data, test_set=test_data).double()
    #model = Regressor_Simple(opt=opt, train_set=train_data, val_set=val_data, test_set=test_data).double()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.device = device
    print(device)
    model = model.to(device)
    # print(model)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min")
    #best_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode="min")

    # Wandb Logger:
    '''
    runname = f'{type(model).__name__}-{datetime.now().strftime("%d/%m/%Y|%H:%M")}-{os.environ.get("USERNAME")}'
    wandb_logger = WandbLogger(project="PredictProtein2", name=runname)
    wandb_logger.experiment.config.update({'protei_embed_type': opt.prot_embedd,
                                           'ligand_embed_type': opt.lig_embedd,
                                           'model_type': type(model).__name__
                                           })
    wandb_logger.watch(model)
    '''

    if opt.hp_tuning:
        '''
        Best trial:
        '''
        pruning = False  # if pruner shall be used
        pruner: optuna.pruners.BasePruner = (
            optuna.pruners.MedianPruner() if pruning else optuna.pruners.NopPruner()
        )
        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(objective, n_trials=opt.trials, timeout=6000)
        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))


    else: # Normal Training
        trainer = pl.Trainer(
            max_epochs=opt.epochs,
            accelerator='gpu' if device == torch.device('cuda') else None,
            devices=1 if device == torch.device('cuda') else None,
            # distributed_backend='dp',
            callbacks=early_stop_callback,
            num_sanity_val_steps=0,
            # logger=True,
            # log_every_n_steps=50,
            # logger=wandb_logger,
            accumulate_grad_batches=opt.acc_grad,
        )
        # trainer.validate(model, dataloaders=val) # only needed when validation curve shall start on x=0
        trainer.fit(model)  # train the standard regressor
        #final_model_path = trainer.checkpoint_callback.best_model_path  # load best model checkpoint
        #result = trainer.validate(ckpt_path=final_model_path,
        #                          dataloaders=val)  # Validate after training and store results
        ckpt = torch.load('lightning_logs/version_' + str(model.logger.version) + '/best_model.pt')
        model.load_state_dict(ckpt, strict=True)
        result = trainer.validate(model, dataloaders=val)
        result_test = trainer.validate(model, dataloaders=test)


def objective(trial):
    # PyTorch Lightning will try to restore model parameters from previous trials if checkpoint
    # filenames match. Therefore, the filenames for each trial must be made unique.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join("/logs", "trial_{}".format(trial.number)), monitor="mae_pts"
    )

    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We create a simple logger instead that holds the log in memory so that the
    # final accuracy can be obtained after optimization. When using the default logger, the
    # final accuracy could be stored in an attribute of the `Trainer` instead.
    # logger = DictLogger(trial.number)

    # Define Hyperparameter ranges:
    opt.lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    #opt.num_layers = trial.suggest_int("num_layers", 1, 6)
    opt.hidden_size = trial.suggest_int("hidden_size", 100, 500)
    #opt.drop = trial.suggest_float("dropout", 0.0, 0.7)
    #opt.epochs = trial.suggest_int("epochs", 10, 200)
    opt.show_scores = False

    trainer = pl.Trainer(
        logger=True,
        max_epochs=opt.epochs,
        gpus=1 if torch.cuda.is_available() else None,
        # callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_accuracy")],
    )

    model = globals()[opt.model](opt=opt, train_set=train_data, val_set=val_data, test_set=test_data).double()
    trainer.fit(model)

    return trainer.callback_metrics["val_mae_risk"].item()

def fix_data(data):

    # transfer back to proper list input types
    data = np.array(data)#[:1000]

    remove_index = []
    for i in range(0,len(data)):
        for j in range(len(data[i])):
            #print(data[i,j])
            #print(i)
            try:
                data[i,j] = ast.literal_eval(str(data[i,j]))
                if j in [6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]:
                    data[i,j] = list(map(float, data[i,j]))
            except ValueError:
                remove_index.append(i)
                continue
    # remove nan values:
    data = list(data)
    for idx in sorted(remove_index, reverse=True):
        del data[idx]
    data = np.array(data)

    # combine sentences:
    texts = []
    for i in range(0, len(data)):
        for j in range(len(data[i,4])):
            if j==0:
                text = data[i,4][j]
            else:
                text += ' '+data[i,4][j]

        text = deEmojify(text)
        texts.append(text)
    data = np.concatenate((data, np.array(texts).reshape(len(data), 1)),axis=1)

    return data

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def split_sets(data):

    data_train, data_testy = train_test_split(data, test_size=0.2,random_state=1)
    data_val, data_test = train_test_split(data_testy, test_size=0.5, random_state=1)

    return data_train, data_val, data_test

if __name__ == '__main__':
    opt = init_parser()
    dataset_path = os.path.join(Path(os.path.dirname(os.path.abspath(os.getcwd()))), "Data/data_"+str(opt.seq_length)+".csv")
    data = fix_data(pd.read_csv(dataset_path)[:5000])
    print(len(data))
    data_train, data_val, data_test = split_sets(data)

    train_data = Dataset(mode="train", X=data_train, opt=opt)
    val_data = Dataset(mode="val", X=data_val, opt=opt)
    test_data = Dataset(mode="test", X=data_test, opt=opt)

    # baseline test:
    print("Mean Value of train labels: "+ str(torch.mean(train_data.y,axis=0)))
    MAE = nn.L1Loss()
    print("Baseline MAE on val set: "+str(MAE(val_data.y, torch.mean(train_data.y,axis=0).repeat(len(val_data)))))

    train_model(opt, train_data, val_data, test_data)

    # baseline test:
    print("Mean Value of train labels: " + str(torch.mean(train_data.y, axis=0)))
    MAE = nn.L1Loss()
    print("Baseline MAE on val set: " + str(MAE(val_data.y, torch.mean(train_data.y, axis=0).repeat(len(val_data)))))