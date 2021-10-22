import numpy as np
import glob
import torch.utils.data
import os
import math
import pandas as pd
from skimage import io, transform, util
from PIL import Image
import torch
import torchvision as vision
from torchvision import transforms, datasets
import random
import gin
from gin.config import _CONFIG
import copy

import timeit

SEED_FIXED = 100000

def load_modelviews(files):
    imgs = []
    for f in files:
        im = np.array(Image.open(f).convert('RGB'))
        imgs.append(im)
    return np.stack(imgs)
            
def load_modelinto_numpy(root_dir, classnames, ending='/*.png', numviews = 12):
    set_ = root_dir.split('/')[-1]
    parent_dir = root_dir.rsplit('/',2)[0]
    filepaths = []
    
    data = []
    labels = []
    
    for i in range(len(classnames)):
        all_files = sorted(glob.glob(parent_dir+'/'+classnames[i]+'/'+set_+ending))
        
        nummodels = int(len(all_files)/12)
        print('Transformming %d models of class %d - %s into tensor'%(nummodels, i, classnames[i]))
        starting_time = timeit.default_timer()
        
        for m_ind in range(nummodels):
            modelimgs = load_modelviews(all_files[m_ind*12:m_ind*12+12])
            #print(all_files[m_ind*12].rsplit('.',2)[0]+'.npy')
            with open(all_files[m_ind*12].rsplit('.',2)[0]+'.npy', 'wb') as f:
                torch.save(modelimgs, f)
        
        print('... finished in %.2fs'%(timeit.default_timer() - starting_time))


class MultiviewModelDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, ending='/*.png',
                 num_views=12, shuffle=True, specific_view=None, transform=None):

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        self.root_dir = root_dir
        
        self.num_views = num_views
        self.specific_view = specific_view

        self.transform = transform
        self.init_filepaths(ending)

    def init_filepaths(self, ending):
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(self.root_dir.rsplit('/',2)[0]+'/'+self.classnames[i]+'/'+self.root_dir.split('/')[-1]+ending))
            files = [] 
            for file in all_files:
                files.append(file.split('.obj.')[0])
                
            files = list(np.unique(np.array(files)))
            self.filepaths.extend(files)
    
    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        imgs = torch.load(path+'.obj.npy')
        trans_imgs = []
        for img, view in zip(imgs[self.specific_view], self.specific_view):
            if self.transform:
                img = self.transform(img)
            trans_imgs.append(img) 
        data = torch.stack(trans_imgs)
        return idx, data, class_id
    

@gin.configurable 
def get_mvdcndata(
        ending = '/*.png',
        root_dir = os.environ['DATA_DIR'], 
        make_npy_files = False,
        valid_size=0.2,
        batch_size=8,
        random_seed_for_validation = 10,
        num_views=12,
        num_workers=0,
        specific_views=None,
        seed=777,
        use_cuda=True,
        ):
    random.seed(seed)
    np.random.seed(seed) # cpu vars
    torch.manual_seed(seed) # cpu  vars
    if use_cuda: torch.cuda.manual_seed_all(seed)
    
    test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]) 

    if make_npy_files:
        classnames = ['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                     'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                     'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                     'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                     'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        load_modelinto_numpy(os.path.join(root_dir, '*/test'), classnames, ending='/*.png', numviews = 12)
        load_modelinto_numpy(os.path.join(root_dir, '*/train'), classnames, ending='/*.png', numviews = 12)

    train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    test_dataset = MultiviewModelDataset(os.path.join(root_dir, '*', 'test'),
        ending=ending,
        num_views=num_views, 
        specific_view=specific_views, 
        transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers)

    training = MultiviewModelDataset(os.path.join(root_dir, '*', 'train'), 
        ending=ending, 
        num_views=num_views, 
        specific_view=specific_views, 
        transform=train_transform)

    num_train = len(training)
    indices = list(range(num_train))
    training_idx = indices

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg
    
    split = int(np.floor(valid_size * num_train))
    random.Random(random_seed_for_validation).shuffle(indices)
    training_idx, valid_idx = indices[split:], indices[:split]
    
    valid_sub = torch.utils.data.Subset(training, valid_idx)
    valid_loader = torch.utils.data.DataLoader(valid_sub,
                       batch_size=batch_size,
                       shuffle=False,
                       num_workers=num_workers,
                       ) 

    training_sub = torch.utils.data.Subset(training, training_idx)

    training_loader = torch.utils.data.DataLoader(training_sub,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers,
                                                   ) 
    
    return training_loader, valid_loader, test_loader





    