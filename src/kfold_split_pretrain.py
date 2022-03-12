import os
import pandas as pd
import numpy as np
from utils import seed_torch
import argparse
import train_configs
def split_kfold(input_csv,output_csv,n_fold=5,n_class_skip=1):
    train = pd.read_csv(input_csv)
    train['classes'] = train.groupby('individual_id').ngroup()
    train['classes_species'] = train.groupby('species').ngroup()
    df_list = []
    for index,df in train.groupby('classes'):
        if len(df)>n_class_skip:
            df = df.sample(frac=1)
            df['fold'] = (np.arange(len(df)) + np.random.randint(n_fold)) % n_fold
            df_list.append(df)
        else:
            df['fold'] = -1
            df_list.append(df) 
    train = pd.concat(df_list)
    for row in range(n_fold):
        print(len(list(set(train[train.fold==row].classes_species))))
    train.to_csv(output_csv,index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='kfold')
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    config = eval(f"train_configs.{args.config}.config")
    
    seed_torch(config.seed)
    split_kfold(config.input_csv,config.kfold_csv,config.n_folds,config.n_class_skip)