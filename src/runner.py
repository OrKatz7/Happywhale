import os
import math
import time
import random
import shutil
import torch
import numpy
from pathlib import Path
from utils import *
from post_processing import *
def train_fn(config,train_loader, model, criterion , optimizer, epoch, scheduler, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0
    for step, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = batch['image'].to(device)
        labels = batch['label'].to(device).long()
        labels2 = batch['species'].to(device).long()
        batch_size = labels.size(0)
        output = model(images,labels)
        loss = config.arcface_w*criterion(output['arcface'], labels) + config.species_w*criterion(output['species'], labels2)
        # record loss 
        losses.update(loss.item(), batch_size)
        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps
        if config.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        if (step + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % config.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  #'LR: {lr:.6f}  '
                  .format(
                   epoch+1, step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(train_loader)),
                   grad_norm=grad_norm,
                   #lr=scheduler.get_lr()[0],
                   ))
    return losses.avg

def emb_fn(valid_loader, model, device):
    model.eval()
    emb = []
    targets = []
    start = end = time.time()
    for step, batch in enumerate(tqdm(valid_loader)):
        # measure data loading time
        images = batch['image'].to(device)
        labels = batch['label'].to(device).long()
        labels2 = batch['species'].to(device).long()
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            y_preds = model.extract_feature(images).cpu().numpy()
            emb.append(y_preds)
            targets.append(labels.cpu().numpy())
    emb = np.concatenate(emb)
    targets = np.concatenate(targets)
    
    return emb,targets

def valid_fn(config,model,train_loader,valid_loader,device,run_cv = True):
    emb_v,targets_v = emb_fn(valid_loader, model, device)
    emb_t,targets_t = emb_fn(train_loader, model, device)
    res = {}
    res['emb_v'] = emb_v
    res['targets_v'] = targets_v
    res['emb_t'] = emb_t
    res['targets_t'] = targets_t
    if not run_cv: 
        return res
    tr_embeddings = res['emb_t']
    val_embeddings = res['emb_v']
    targets = res['targets_v']
    targets_train = res['targets_t']
    EMB_SIZE = 512
    vals_blend = []
    labels_blend = []
    inds_blend = []
    for i in range(1):
        vals, inds = get_topk_cossim_sub(val_embeddings[:,i*EMB_SIZE:(i+1)*EMB_SIZE], tr_embeddings[:,i*EMB_SIZE:(i+1)*EMB_SIZE], k=500)
        vals = vals.data.cpu().numpy()
        inds = inds.data.cpu().numpy()
        labels = np.concatenate([targets_train[inds[:,i]].reshape(-1,1) for i in range(inds.shape[1])], axis=1)

        vals_blend.append(vals)
        labels_blend.append(labels)
        inds_blend.append(inds)
    vals = np.concatenate(vals_blend, axis=1)
    inds = np.concatenate(inds_blend, axis=1)
    labels = np.concatenate(labels_blend, axis=1)
    M = []
    for row in range(len(vals)):
        m = map_per_image(targets[row],labels[row])
        M.append(m)
    return np.array(M).mean()

def valid_fn_acc(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to evaluation mode
    model.eval()
    preds = []
    preds2 = []
    start = end = time.time()
    for step, batch in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = batch['image'].to(device)
        labels = batch['label'].to(device).long()
        labels2 = batch['species'].to(device).long()
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            y_preds = model(images,labels)
        loss = 0.05*criterion(y_preds[0], labels) + 0.05*criterion(y_preds[1], labels2) + 0.9*criterion(y_preds[2], labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(y_preds[0].softmax(1).to('cpu').numpy())
        preds2.append(y_preds[1].softmax(1).to('cpu').numpy())
        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % config.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(
                   step, len(valid_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(valid_loader)),
                   ))
    predictions = np.concatenate(preds)
    predictions2 = np.concatenate(preds2)
    return losses.avg, predictions,predictions2
