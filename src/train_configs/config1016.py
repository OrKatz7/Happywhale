import albumentations
from albumentations.pytorch import ToTensorV2
class config:
    exp_name = 'exp1016'
    seed = 42
    epochs = 25
    kfold_csv = '/home/notebooks/Happywhale/input/fold.csv'
    n_folds = 5
    log_DIR = '/home/notebooks/Happywhale/log/'
    save_dir = '/home/notebooks/Happywhale/output/'
    load_from = [None,None,None,None,None]
    debug=False
    apex=True
    print_freq=100
    val_freq = 3
    val_epoch_freq = 8
    size=768
    gradient_accumulation_steps=1
    max_grad_norm=1000
    ### data
    input_csv = '/home/nvme/happy-whale-and-dolphin/train.csv'
    data_root_path = '/home/nvme/happy-whale-and-dolphin/train_images/'
    batch_size = 16
    num_workers = 4
    n_class_skip = 1
    device = "cuda:0"
    arcface_w = 0.5
    species_w = 0.1
    id_w = 0.4
    swa = False
    swa_start =16
    swa_lr = 1e-3
    crop_p = 0.999
    crop_csv_path = '/home/notebooks/Happywhale/input/train_bbox.csv' #new yolo
    crop_backfin_csv_path = '/home/notebooks/Happywhale/input/train_bbox_backfin.csv'
    use_crop_for_val = True
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
    criterion = [{'f_class': "torch.nn.CrossEntropyLoss","args": {},'name':"arcface" , 'w':0.5,"label":"label"},
                {'f_class': "torch.nn.CrossEntropyLoss","args": {},'name':"species", 'w':0.1,"label": "species"},
                {'f_class': "torch.nn.CrossEntropyLoss","args": {},'name':"u_id",'w':0.1,"label":"label"},
                {'f_class': "models.triplet_loss.TripletLoss","args": {"margin":0.3},'name':"feature", 'w':0.3,"label": "label"},
                ]
    
    train_transforms = albumentations.Compose([
            albumentations.Resize(size//2, size),
            albumentations.HorizontalFlip(p=0.5),
            # albumentations.RandomBrightness(limit=0.2, p=0.25),
            # albumentations.RandomContrast(limit=0.2, p=0.25),
            albumentations.ImageCompression(quality_lower=95, quality_upper=100),
            # albumentations.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=0, p=0.25),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.5),
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
