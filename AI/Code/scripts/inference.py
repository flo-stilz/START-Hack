# Evaluation script
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from dataset.Dataset import *
from models.regressor_models import Regressor_Simple
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, BertTokenizer


def init_parser():
    # Settings
    parser = argparse.ArgumentParser(
        description='Hyperparameters for Protein Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', type=int, required=False, default=1,
                        help='Batch size!')
    parser.add_argument('--lr', type=float, required=False, default=1e-3,
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
    parser.add_argument('--standardize', action='store_true',
                        help='standardize data')
    parser.add_argument('--version', type=str, required=True,
                        help='Enter version number from the desired model')
    parser.add_argument('--no_ckpt', action='store_false',
                        help='save model ckpt')
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

def find_path(opt):


    parent_path = os.path.join(os.getcwd(), "lightning_logs/version_" + opt.version)
    model_path = os.path.join(parent_path, "best_model.pt")

    return model_path, opt

def eval_model(opt, model_path):

    # Initialize model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

    model = globals()[opt.model](opt=opt, train_set=None, val_set=None, test_set=None).double()
    #model = Regressor_Simple(opt=opt, train_set=train_data, val_set=val_data, test_set=test_data).double()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.device = device
    print(device)
    model = model.to(device)

    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    samples = np.array(['thats the one I want mega miner a dawg damn everyone getting him now huh Yuh New emote I’m better Nobody has more kisses than me You’re probably better than me But Nah you’re definitely better I used my stupid sparky challenge deck Lol E-giant and Ebarbs. That’s nastyyyy lol GG Lol I know I’m a bitch But it works Nah you’re cool lol. I’ve got a lot of experience playing them in older ladder so I’ve been there lol.Daddy I mean yeah I hate the mega knight Autocorrect is crazy Shitttttt don’t threaten me with a good time lol That’s what I’m gonna do to your feet Say lessssss Jstar stop dick riding the leader you fag Jstar a hotdog rider Meat grinder Stop sucking up Tbh i didn’t know he was leader',
                        'Ok then pu$$y I was playing for fun and u lost Yo afraid to go for round 2 I wasn’t trying you dumb fuck Nah u afraid to talk to a women Wasn’t trying I banged your mom I wish I used a condom Only part that sucked was I had to wait in line first I wish I never met with your mother A disgrace child That ugly twat gave me 10 sti’s My kcok itches I don’t even know if I am your father because your mom has many',
                        'Ummm it’s a shit deck ofc you can’t make it work, Kys yes I can get cooked shut up you use classic logbait you’re just mad bc you can’t make an original deck It works dosent it why do I need original deck if I win with this one using a deck just because it works is honestly so bland and it’s people like you that make the game boring I have autism and just want to have fun Yoooo that’s sooo fucked up you making fun of someone with autos haha he has autism what a loser skill issue tbh',
                        'already using a negative IQ deck dont be such a shame',
                        'looser suck it my son such it dont be shy with this level, baby can play this',
                        'noob you behave like an asshole"looks like at all good deck good game you can leave now its just a game',
                        'wassup have you seen the swastika in my profile? nah bruh dammn, thats deep aight brb, cya'])

    pre_tok_samples = pre_tokenize_text(samples)
    tokenizer = AutoTokenizer.from_pretrained(opt.bert_model, add_prefix_space=True)
    encodings = tokenizer(list(pre_tok_samples), is_split_into_words=True, return_offsets_mapping=True, padding=True,truncation=True, max_length=512)
    encodings.pop("offset_mapping")
    input_ids = torch.tensor(encodings['input_ids']).to(device)
    attention_masks = torch.tensor(encodings['attention_mask']).to(device)

    pred = model(input_ids, attention_masks)
    print("Model Predictions:")
    print(pred)

def pre_tokenize_text(data):
    data = np.array(data)
    data_pre_tok = []
    for i in range(0, len(data)):
        doc = word_tokenize(data[i])  # find correct index
        data_pre_tok.append(doc)
    return data_pre_tok



if __name__ == '__main__':
    opt = init_parser()
    model_path, opt = find_path(opt)

    eval_model(opt, model_path)
