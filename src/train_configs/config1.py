import albumentations
from albumentations.pytorch import ToTensorV2
class config:
    exp_name = 'exp1'
    seed = 42
    epochs = 20
    kfold_csv = '/home/kaor/whale_code/csv/fold.csv'
    n_folds = 5
    log_DIR = '/home/kaor/whale_code/log/'
    save_dir = '/home/kaor/whale_code/output/'
    load_from = [None,None,None,None,None]
    debug=False
    apex=False
    print_freq=100
    size=512
    gradient_accumulation_steps=1
    max_grad_norm=1000
    ### data
    input_csv = '/home/kaor/whale/train.csv'
    data_root_path = '/home/kaor/whale/train_images/'
    batch_size = 12
    num_workers = 4
    n_class_skip = 1
    device = "cuda:0"
    arcface_w = 0.8
    species_w = 0.2
    ### torch
    model = {'f_class': "models.eff.get_model",
             "args": {
                     'backbone_name':'tf_efficientnet_b5_ns',
                     'num_calss_id':15587,
                     'num_calss':30}}
    optimizer = {'f_class': "torch.optim.Adam",
             "args": {
                     'lr':1e-4,
                     'weight_decay':1e-6,
                     'amsgrad':False}}
    scheduler = {'f_class': "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
             "args": {
                     'T_0':10,
                     'T_mult':1,
                     'eta_min':1e-7,
                     'last_epoch':-1}}
    criterion = {'f_class': "torch.nn.CrossEntropyLoss",
             "args": {}}
    
    train_transforms = albumentations.Compose([
            albumentations.Resize(512, 512),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightness(limit=0.2, p=0.25),
            albumentations.RandomContrast(limit=0.2, p=0.25),
            albumentations.ImageCompression(quality_lower=99, quality_upper=100),
            albumentations.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=0, p=0.25),
            albumentations.ShiftScaleRotate(shift_limit=0.065, scale_limit=0.1, rotate_limit=360, border_mode=0, p=0.25),
            albumentations.Cutout(max_h_size=int(512 * 0.25), max_w_size=int(512 * 0.25), num_holes=1, p=0.5),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    val_transforms = albumentations.Compose([
                albumentations.Resize(512, 512),
                albumentations.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ])