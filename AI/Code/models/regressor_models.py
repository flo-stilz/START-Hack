# Regressor Simple
import torch
from torch import nn
import pytorch_lightning as pl
from models.regressor_base import *
from transformers import BertModel, DistilBertModel, AutoModel, AutoConfig, AutoModelForSequenceClassification


class Bert(nn.Module):

    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        # set Bert
        configuration = AutoConfig.from_pretrained(self.opt.bert_model)
        self.bert = AutoModel.from_pretrained(self.opt.bert_model, config=configuration)

        # freeze some of the BERT weights:
        if self.opt.freeze_emb and "distil" in self.opt.bert_model:
            modules = [self.bert.embeddings, *self.bert.transformer.layer[:self.opt.freeze]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        elif self.opt.freeze_emb:
            modules = [self.bert.embeddings, *self.bert.encoder.layer[:self.opt.freeze]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, mask):
        # feed x into BERT model!
        if "distilbert-base" in self.opt.bert_model:
            # For DistilBERT
            hidden_state = self.bert(input_ids=input_ids, attention_mask=mask, return_dict=False)
            output = hidden_state[0][:, 0]
        else:
            # For BERT uses final hidden_state for now
            hidden_state, pooled_output = self.bert(input_ids=input_ids, attention_mask=mask, return_dict=False)
            # output = pooled_output
            if self.opt.all_emb:
                output = hidden_state[:, :]
            else:
                output = hidden_state[:, 0]

        return output

class LinearSkip(pl.LightningModule):
    def __init__(self, opt=None):
        super().__init__()

        self.layer = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.act = nn.LeakyReLU()
        self.layer2 = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.act2 = nn.LeakyReLU()

    def forward(self, inputs):

        x_intermediate = self.act(self.layer(inputs))
        x_final = self.layer2(x_intermediate)
        # Skip Connection:
        x = self.act2(x_final + inputs)

        return x


class Regressor_Simple(Regressor_Base):

    def __init__(self, opt=None, train_set=None, val_set=None, test_set=None):
        super().__init__(opt=opt, train_set=train_set, val_set=val_set, test_set=test_set)
        
        self.bert = Bert(opt)
        self.input_size = 768
        self.modules = []
        self.modules.append(nn.Linear(self.input_size, opt.hidden_size))
        self.modules.append(nn.Dropout(opt.drop))
        self.modules.append(nn.LeakyReLU())
        for i in range(1,opt.num_layers-1):
            if opt.use_linear_skip:
                self.modules.append(LinearSkip(opt))
            else:
                self.modules.append(nn.Linear(opt.hidden_size, opt.hidden_size))
                self.modules.append(nn.Dropout(opt.drop))
                self.modules.append(nn.LeakyReLU())
            
        self.modules.append(nn.Linear(opt.hidden_size, self.opt.num_labels))
        
        self.regressor = nn.Sequential(*self.modules)
        
    def forward(self, input_ids, attention_masks):
        
        # final regressor:
        x = self.bert(input_ids, attention_masks)
        x = self.regressor(x)

        return x