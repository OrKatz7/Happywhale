import albumentations
from albumentations.pytorch import ToTensorV2
from augmix import RandomAugMix
class config:
    exp_name = 'exp16v6AugmixFullB4pseudo'
    seed = 42
    epochs = 5
    start_epoch = 0
    kfold_csv = '/home/kaor/whale_code/csv/fold2_old.csv'
    kfold_csv_pseudo = '/home/kaor/whale_code/csv/fold2_pseudo.csv'
    n_folds = 5
    trn_folds = [0,1,2,3,4]
    log_DIR = '/home/kaor/whale_code/log/'
    save_dir = '/sise/liorrk-group/OrDanOfir/output/'
    # load_from = ['/home/kaor/whale_code/output/exp16pretrainb4_fold0_best.pth'] * 5
    load_from = [f'/sise/liorrk-group/OrDanOfir/output/exp16v6AugmixB4Full_fold{row}_best.pth' for row in range(5)]
    # load_from = ['/sise/liorrk-group/OrDanOfir/output/exp16v6AugmixFull_fold0_best.pth'] +  ['/home/kaor/whale_code/output/exp16pretrain_fold0_best.pth'] * 4
    old_model_head_dim = 15587
    debug=False
    apex=True
    print_freq=100
    val_freq = 3
    val_epoch_freq = 10
    size=1024
    gradient_accumulation_steps=1
    max_grad_norm=1000
    ### data
    sampler = False
    input_csv = '/home/kaor/whale/train.csv'
    data_root_path = '/home/kaor/whale/train_images/'
    batch_size = 16
    num_workers = 6
    n_class_skip = 1
    device = "cuda:0"
    arcface_w = 0.5
    species_w = 0.1
    id_w = 0.4
    swa = False
    swa_start =16
    swa_lr = 1e-3
    crop_p = 0.9999
    crop_csv_path = '/sise/home/kaor/whale_code/yolo/bbox_full_train.csv' #new yolo
    crop_backfin_csv_path = '/sise/home/kaor/whale_code/yolo/bbox_full_train.csv'
    use_crop_for_val = True
    ### torch
    model = {'f_class': "models.eff.get_model",
             "args": {
                     'backbone_name':'tf_efficientnet_b4_ns',
                     'num_calss_id':15587,
                     'num_calss':30}}
    optimizer = {'f_class': "torch.optim.Adam",
             "args": {
                     'lr':5e-5,
                     'weight_decay':1e-6,
                     'amsgrad':False}}
    scheduler = {'f_class': "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
             "args": {
                     'T_0':10,
                     'T_mult':1,
                     'eta_min':1e-7,
                     'last_epoch':-1}}
    
    criterion = [{'f_class': "torch.nn.CrossEntropyLoss","args": {},'name':"arcface" , 'w':0.9,"label":"label"},
                {'f_class': "torch.nn.CrossEntropyLoss","args": {},'name':"u_id",'w':0.1,"label":"label"},
                ]
    
    train_transforms = albumentations.Compose([
            albumentations.Resize(size//2, size),
            albumentations.OneOf([
                    RandomAugMix(severity=3, width=3, alpha=1., p=0.75),
                    albumentations.ImageCompression(quality_lower=95, quality_upper=100),
                    albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.75),
                ], p=1.0),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.Cutout(max_h_size=int(size//2 * 0.25), max_w_size=int(size * 0.25), num_holes=1, p=0.5),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    val_transforms = albumentations.Compose([
                albumentations.Resize(size//2, size),
                albumentations.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ])