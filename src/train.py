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
import train_configs
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
                                  crop_backfin_csv_path = config.crop_backfin_csv_path,
                                  use_crop_for_val = config.use_crop_for_val,
                                  singles_in_fold=hasattr(config,'singles_in_fold') and config.singles_in_fold,
                                  singles_in_train=hasattr(config,'singles_in_train') and config.singles_in_train,
                                  min_num_in_train=config.min_num_in_train if hasattr(config,'singles_in_train') else 1,
                                  train_not_seen=hasattr(config,'train_not_seen') and config.train_not_seen,
								  use_sampler = config.sampler)
        
    model = eval(config.model['f_class'])(**config.model['args']).to(config.device)
    if config.load_from[fold] is not None:
        model.reset_head(config.old_model_head_dim)
        model.load_state_dict(torch.load(config.load_from[fold])['model'])
        model.reset_head(config.model['args']['num_calss_id'])
        model.to(config.device)
        print(f"load {config.load_from[fold]}")
        
    optimizer = eval(config.optimizer['f_class'])(model.parameters(),**config.optimizer['args'])
    scheduler = eval(config.scheduler['f_class'])(optimizer,**config.scheduler['args'])
    criterion = {}
    for c in config.criterion:
        criterion[c['name']] = {"criterion": eval(c['f_class'])(**c['args']), 'w':c['w'],"label":c['label']}
    # criterion = eval(config.criterion['f_class'])(**config.criterion['args'])
    scaler = amp.GradScaler()
    if config.swa:
        swa_start = config.swa_start
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=config.swa_lr)
    else:
        swa_start = config.epochs+1
    best_score = 0.
    best_loss = np.inf
    for epoch in range(config.epochs):
        data['train_loader'].sampler.set_epoch(epoch)
        start_time = time.time()
        avg_loss = train_fn(config,data['train_loader'], model, criterion , optimizer, epoch, scheduler, config.device,scaler)
        if epoch > swa_start and config.swa:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        if epoch % (config.val_freq) == 0 or epoch >= config.val_epoch_freq:
            score,score_softmax = valid_fn(config,model,data['train_loader_emb'],data['valid_loader'],device=config.device,
                                           run_cv = True,use_new=hasattr(config,'csv_use_new') and config.csv_use_new,
                                           use_species=hasattr(config,'valid_use_species') and config.valid_use_species)
            LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}')
            LOGGER.info(f'Epoch {epoch+1} - metric top 5 arcface: {score}, metric top 5 softmax {score_softmax}')
            if score > best_score:
                best_score = score
                llr = optimizer.param_groups[0]['lr']
                LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f}, lr: {llr}')
                torch.save({'model': model.state_dict(), },
                                    config.save_dir+f'{config.exp_name}_fold{fold}_best.pth')
    if config.swa:
        update_bn(data['train_loader_emb'], swa_model,device=config.device)
        score = valid_fn(config,swa_model.module,data['train_loader_emb'],data['valid_loader'],device=config.device,
                         run_cv = True, use_new=hasattr(config,'csv_use_new') and config.csv_use_new,
                         use_species=hasattr(config,'valid_use_species') and config.valid_use_species)
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
    print(args.config)
    config = eval(f"train_configs.{args.config}.config")
    seed_torch(config.seed)
    main(config)

    



            
    