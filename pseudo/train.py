import torch
import os
import pandas as pd
import numpy as np
import argparse
from runner import train_fn,emb_fn,valid_fn,valid_fn_acc
from datasets import GetTrainDataLoader,defult_train_trms,defult_val_trms
from utils import seed_torch,init_logger,timer
import albumentations
import models
import time
from swa_utils import AveragedModel, SWALR,update_bn
import torch.cuda.amp as amp
import scheduler
import optim
from timm.utils import ModelEma
from timm.utils import get_state_dict
from torch import distributed

try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )
    
def debug_mode(config,folds):
    if config.debug:
        folds = folds.sample(n=1000, random_state=config.seed).reset_index(drop=True)
        config.epochs = 4
        config.n_folds = 1
    return config,folds

def train_one_fold(config,LOGGER,folds,fold=0):
    data = GetTrainDataLoader(folds=folds,
                                  fold=fold,
                                  train_transforms=config.train_transforms,
                                  val_transforms=config.val_transforms,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  data_root_path=config.data_root_path,
                                  crop_p = config.crop_p,
                                  crop_csv_path = config.crop_csv_path,
                                  crop_backfin_csv_path = config.crop_backfin_csv_path,
                                  use_crop_for_val = config.use_crop_for_val,
                                  use_sampler = config.sampler,
                                  pseudo_csv = config.kfold_csv_pseudo)
        
    model = eval(config.model['f_class'])(**config.model['args']).to(config.device)
    if not config.swa:
        swa_start = config.epochs+1
    else:
        swa_start = config.swa_start
    if config.load_from[fold] is not None:
        from mmcv.runner import load_checkpoint
        torch.save(torch.load(config.load_from[fold])['model'],"temp.pth")
        load_checkpoint(model, "temp.pth")
        # model.reset_head(config.old_model_head_dim)
        # model.load_state_dict(torch.load(config.load_from[fold])['model'])
        # model.reset_head(config.model['args']['num_calss_id'])
        # model.to(config.device)
        # print(f"load {config.load_from[fold]}")
        
    optimizer = eval(config.optimizer['f_class'])(model.parameters(),**config.optimizer['args'])
    scheduler = eval(config.scheduler['f_class'])(optimizer,**config.scheduler['args'])
    criterion = {}
    for c in config.criterion:
        criterion[c['name']] = {"criterion": eval(c['f_class'])(**c['args']), 'w':c['w'],"label":c['label'],"epoch":True if "epoch" in c else False}
    # criterion = eval(config.criterion['f_class'])(**config.criterion['args'])
    scaler = amp.GradScaler()
    best_score = 0.
    best_loss = np.inf
    for epoch in range(config.start_epoch,config.epochs):
        # if config.swa and epoch==swa_start:
        #     model_ema = ModelEma(model,decay=0.95)
        if config.sampler:
            data['train_loader'].sampler.set_epoch(epoch)
        start_time = time.time()
        avg_loss = train_fn(config,data['train_loader'], model, criterion , optimizer, epoch, scheduler, config.device,scaler)
        scheduler.step()
        llr = optimizer.param_groups[0]['lr']
        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}, lr: {llr}')
        # if epoch > config.swa_start and config.swa:
        # model_ema.update(model)
        if epoch % (config.val_freq) == 0 or epoch >= config.val_epoch_freq:
            score,score_softmax = valid_fn(config,model,data['train_loader_emb'],data['valid_loader'],device=config.device,run_cv = True)
            
            
            LOGGER.info(f'Epoch {epoch+1} - metric top 5 arcface: {score}, metric top 5 softmax {score_softmax}')
            if score > best_score:
                best_score = score
                
                LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f}, lr: {llr}')
                torch.save({'model': model.state_dict(), },
                                    config.save_dir+f'{config.exp_name}_fold{fold}_best.pth')
                # if epoch > swa_start and config.swa:
                #     model_ema.update(model)
    # if config.swa:
    #     # update_bn(data['train_loader_emb'], swa_model,device=config.device)
    #     score = valid_fn(config,model_ema.ema ,data['train_loader_emb'],data['valid_loader'],device=config.device,run_cv = True)
    #     LOGGER.info(f'SWA - metric top 5: {score}')
    #     torch.save({'model': get_state_dict(model_ema), },
    #                                 config.save_dir+f'{config.exp_name}_fold{fold}_swa.pth')
def main(config):
    LOGGER = init_logger(log_file=f'{config.log_DIR}/{config.exp_name}.log')
    folds = pd.read_csv(config.kfold_csv)
    config,folds = debug_mode(config,folds)
    for fold in range(config.n_folds):
        train_one_fold(config,LOGGER,folds,fold)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train pipeline')
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    config = eval(args.config)
    seed_torch(config.seed)
    main(config)