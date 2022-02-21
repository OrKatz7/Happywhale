import os
import pandas as pd
import numpy as np
from torch_utils import seed_torch

def split_kfold(input_csv,output_csv,n_fold):
    train = pd.read_csv(input_csv)
    train['classes'] = train.groupby('individual_id').ngroup()
    train['classes_species'] = train.groupby('species').ngroup()
    df_list = []
    for index,df in train.groupby('classes'):
        if len(df)>1:
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
    seed_torch(0)
    n_fold=5
    input_csv = '../whale/train.csv'
    output_csv = "../whale/folds.csv"
    split_kfold(input_csv,output_csv,n_fold)
#     classes = list(set(train.individual_id))
#     classes_species = list(set(train.species))
#     sorted(classes)
#     sorted(classes_species)
#     print(train.head())
#     print(len(classes))
#     print(len(classes_species))
