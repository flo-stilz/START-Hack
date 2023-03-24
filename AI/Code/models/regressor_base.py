# NBA Player Prediction models
import pytorch_lightning as pl
import torch
import sys
from torch import nn
from utils.metrics import concordance_index,get_rm2

class Regressor_Base(pl.LightningModule):

    def __init__(self, opt=None, train_set=None, val_set=None, test_set=None):
        super().__init__()
        self.opt = opt
        self.batch_size = opt.batch_size
        self.data = {'train': train_set,
                     'val': val_set,
                     'test': test_set}
        self.save_hyperparameters()
        self.best_mae = 1000000 # init with arbitrary high value
        self.lr, self.higher_loss_count, self.best_prev_loss = self.opt.lr, 0, 10000000

    def general_step(self, batch, batch_idx, mode):

        input_ids = batch['input_ids']
        attention_masks = batch['attention_mask']
        gt = batch['labels']
        gt = gt.reshape(gt.shape[0], self.opt.num_labels)

        out = self.forward(input_ids, attention_masks)

        # loss
        loss_func = nn.MSELoss()
        mse_loss = 0
        for i in range(self.opt.num_labels):
            mse_loss += loss_func(out[:,i], gt[:,i])

        MAE = nn.L1Loss()
        mae_risk = MAE(out[:,0].reshape(len(out), 1), gt[:,0].reshape(len(gt),1))
        #mae_risk = MAE(out, gt)
        if self.opt.num_labels>1:
            mae_assist = MAE(out[:,1].reshape(len(out),1), gt[:,1].reshape(len(gt),1))
        else:
            mae_assist = torch.tensor(0.0).to(self.opt.device)
        if mode == 'val':
            return mse_loss,mae_risk,mae_assist,out[:,0].reshape(len(out), 1),gt[:,0].reshape(len(gt),1),

        return mse_loss, mae_risk, mae_assist

    def general_end(self, outputs, mode):
        #print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated() / 1024 / 1024 / 1024))
        #print("torch.cuda.memory_cached: %fGB" % (torch.cuda.memory_cached() / 1024 / 1024 / 1024))
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_mse_loss'] for x in outputs]).mean()
        mae_risk = torch.stack([x[mode + '_mae_risk'] for x in outputs]).mean()
        mae_assist = torch.stack([x[mode + '_mae_assist'] for x in outputs]).mean()

        if self.opt.show_scores:
            predictions = torch.flatten(torch.stack([x['y_hat'] for x in outputs]))
            ground_truth = torch.flatten(torch.stack([x['y_true'] for x in outputs]))

            ci = concordance_index(y_true=ground_truth.cpu().numpy(),y_preds=predictions.cpu().numpy())
            r2m = get_rm2(y_true=ground_truth.cpu().numpy(),y_hat=predictions.cpu().numpy())
        else:
            ci=torch.tensor(0)
            r2m=torch.tensor(0)
        return avg_loss, mae_risk, mae_assist, ci,r2m

    def training_step(self, batch, batch_idx):
        mse_loss, mae_risk, mae_assist = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': mse_loss}
        should_log = True
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            should_log = (
                    (self.global_step + 1) % self.opt.acc_grad == 0
            )
        if should_log:
            # self.logger.experiment.add_scalar("train_loss", mse_loss, self.global_step)
            self.log("train_mse_loss", mse_loss)

        return {'loss': mse_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, mae_risk, mae_assist, preds, gt = self.general_step(batch, batch_idx, "val")

        return {'val_mse_loss': loss, 'val_mae_risk': mae_risk,'val_mae_assist': mae_assist,
                'y_hat':preds,'y_true':gt}

    def test_step(self, batch, batch_idx):
        loss, mae_risk, mae_assist = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_mae_risk': mae_risk, 'test_mae_assist': mae_assist}

    def validation_epoch_end(self, outputs):
        avg_loss, mae_risk, mae_assist, ci, r2m = self.general_end(outputs, "val")
        print("Val-Loss={}".format(avg_loss))
        print("Val-MAE-Risk={}".format(mae_risk))
        print("Val-MAE-Assist={}".format(mae_assist))
        print(f'Concordance Index= {ci}')
        print(f'R2 modified= {r2m}')
        self.log("val_loss", avg_loss)
        self.log("val_mae_risk", mae_risk)
        self.log("val_mae_assist", mae_assist)
        self.log("val_ci",ci)
        self.log("val_r2m",r2m)
        if self.opt.adj_lr:
            self.adjust_learning_rate(avg_loss)
        if mae_risk < self.best_mae and not self.opt.no_ckpt:
            torch.save(self.state_dict(), 'lightning_logs/version_'+str(self.logger.version)+'/best_model.pt')
            self.best_mae =mae_risk


        tensorboard_logs = {'val_loss': avg_loss, 'val_mae_risk': mae_risk, 'val_mae_assist':mae_assist}
        return {'val_loss': avg_loss, 'val_mae_risk': mae_risk, 'val_mae_assist': mae_assist, 'log': tensorboard_logs}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=self.batch_size, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['val'], shuffle=False, batch_size=self.batch_size, drop_last=True, pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['test'], shuffle=False, batch_size=self.batch_size)

    def configure_optimizers(self):

        params = self.parameters()

        optim = torch.optim.Adam(params=params, betas=(0.9, 0.999), lr=self.opt.lr, weight_decay=self.opt.reg)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.opt.gamma)

        return [optim]#, [scheduler]

    def adjust_learning_rate(self, avg_loss):

        if avg_loss >= self.best_prev_loss:
            self.higher_loss_count += 1
            if self.higher_loss_count >= 10:
                self.lr = self.lr / 10
                self.trainer.optimizers[0].param_groups[0]['lr'] = self.lr
                self.higher_loss_count = 0
        else:
            self.higher_loss_count = 0
            self.best_prev_loss = avg_loss