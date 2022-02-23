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
def debug_mode(config,folds):
    if config.debug:
        folds = folds.sample(n=1000, random_state=config.seed).reset_index(drop=True)
        config.epoch = 10
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
                                  use_crop_for_val = config.use_crop_for_val)
        
    model = eval(config.model['f_class'])(**config.model['args']).to(config.device)
    if config.load_from[fold] is not None:
        model.load_state.dict(torch.load(config.load_from[fold])['model'])
    optimizer = eval(config.optimizer['f_class'])(model.parameters(),**config.optimizer['args'])
    scheduler = eval(config.scheduler['f_class'])(optimizer,**config.scheduler['args'])
    criterion = eval(config.criterion['f_class'])(**config.criterion['args'])
    if config.swa:
        swa_start = config.swa_start
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=config.swa_lr)
    else:
        swa_start = config.epochs+1
    best_score = 0.
    best_loss = np.inf
    for epoch in range(config.epochs):
        start_time = time.time()
        avg_loss = train_fn(config,data['train_loader'], model, criterion , optimizer, epoch, scheduler, config.device)
        if epoch > swa_start and config.swa:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        if epoch % (config.val_freq) == 0 or epoch >= config.val_epoch_freq:
            score = valid_fn(config,model,data['train_loader_emb'],data['valid_loader'],device=config.device,run_cv = True)
            # if isinstance(scheduler, ReduceLROnPlateau):
            #     scheduler.step(avg_val_loss)
            # else:
            LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}')
            LOGGER.info(f'Epoch {epoch+1} - metric top 5: {score}')
            if score > best_score:
                best_score = score
                LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f}')
                torch.save({'model': model.state_dict(), },
                                    config.save_dir+f'{config.exp_name}_fold{fold}_best.pth')
    if config.swa:
        update_bn(data['train_loader_emb'], swa_model,device=config.device)
        score = valid_fn(config,swa_model.module,data['train_loader_emb'],data['valid_loader'],device=config.device,run_cv = True)
        LOGGER.info(f'SWA - metric top 5: {score}')
        torch.save({'model': swa_model.module.state_dict(), },
                                    config.save_dir+f'{config.exp_name}_fold{fold}_swa.pth')
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

    



            
    
