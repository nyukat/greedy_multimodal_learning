import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import glob
import gin
from gin.config import _CONFIG

from src.utils import numpy_to_torch, torch_to

@gin.configurable
class MMTM_mitigate(nn.Module):
    def __init__(self, 
            dim_visual, 
            dim_skeleton, 
            ratio, 
            device=0,
            SEonly=False, 
            shareweight=False):
        super(MMTM_mitigate, self).__init__()
        dim = dim_visual + dim_skeleton
        dim_out = int(2*dim/ratio)
        self.SEonly = SEonly
        self.shareweight = shareweight

        self.running_avg_weight_visual = torch.zeros(dim_visual).to("cuda:{}".format(device))
        self.running_avg_weight_skeleton = torch.zeros(dim_visual).to("cuda:{}".format(device))
        self.step = 0

        if self.SEonly:
            self.fc_squeeze_visual = nn.Linear(dim_visual, dim_out)
            self.fc_squeeze_skeleton = nn.Linear(dim_skeleton, dim_out)
        else:    
            self.fc_squeeze = nn.Linear(dim, dim_out)

        if self.shareweight:
            assert dim_visual == dim_skeleton
            self.fc_excite = nn.Linear(dim_out, dim_visual)
        else:
            self.fc_visual = nn.Linear(dim_out, dim_visual)
            self.fc_skeleton = nn.Linear(dim_out, dim_skeleton)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, 
                visual, 
                skeleton, 
                return_scale=False,
                return_squeezed_mps = False,
                turnoff_cross_modal_flow = False,
                average_squeezemaps = None,
                curation_mode=False,
                caring_modality=0,
                ):

        if self.SEonly:
            tview = visual.view(visual.shape[:2] + (-1,))
            squeeze = torch.mean(tview, dim=-1)
            excitation = self.fc_squeeze_visual(squeeze)
            vis_out = self.fc_visual(self.relu(excitation))

            tview = skeleton.view(skeleton.shape[:2] + (-1,))
            squeeze = torch.mean(tview, dim=-1)
            excitation = self.fc_squeeze_skeleton(squeeze)
            sk_out = self.fc_skeleton(self.relu(excitation))

        else:
            if turnoff_cross_modal_flow:

                tview = visual.view(visual.shape[:2] + (-1,))
                squeeze = torch.cat([torch.mean(tview, dim=-1), 
                    torch.stack(visual.shape[0]*[average_squeezemaps[1]])], 1)
                excitation = self.relu(self.fc_squeeze(squeeze))

                if self.shareweight:
                    vis_out = self.fc_excite(excitation)
                else:
                    vis_out = self.fc_visual(excitation)

                tview = skeleton.view(skeleton.shape[:2] + (-1,))
                squeeze = torch.cat([torch.stack(skeleton.shape[0]*[average_squeezemaps[0]]), 
                    torch.mean(tview, dim=-1)], 1)
                excitation = self.relu(self.fc_squeeze(squeeze))
                if self.shareweight:
                    sk_out = self.fc_excite(excitation)
                else:
                    sk_out = self.fc_skeleton(excitation)

            else: 
                squeeze_array = []
                for tensor in [visual, skeleton]:
                    tview = tensor.view(tensor.shape[:2] + (-1,))
                    squeeze_array.append(torch.mean(tview, dim=-1))

                squeeze = torch.cat(squeeze_array, 1)
                excitation = self.fc_squeeze(squeeze)
                excitation = self.relu(excitation)

                if self.shareweight:
                    sk_out = self.fc_excite(excitation)
                    vis_out = self.fc_excite(excitation)
                else:
                    vis_out = self.fc_visual(excitation)
                    sk_out = self.fc_skeleton(excitation)

        vis_out = self.sigmoid(vis_out)
        sk_out = self.sigmoid(sk_out)

        self.running_avg_weight_visual = (vis_out.mean(0) + self.running_avg_weight_visual*self.step).detach()/(self.step+1)
        self.running_avg_weight_skeleton = (vis_out.mean(0) + self.running_avg_weight_skeleton*self.step).detach()/(self.step+1)
        
        self.step +=1

        if return_scale:
            scales = [vis_out.cpu(), sk_out.cpu()]
        else:
            scales = None

        if return_squeezed_mps:
            squeeze_array = [x.cpu() for x in squeeze_array]
        else:
            squeeze_array = None

        if not curation_mode:
            dim_diff = len(visual.shape) - len(vis_out.shape)
            vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

            dim_diff = len(skeleton.shape) - len(sk_out.shape)
            sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)
        
        else:
            if caring_modality==0:
                dim_diff = len(skeleton.shape) - len(sk_out.shape)
                sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

                dim_diff = len(visual.shape) - len(vis_out.shape)
                vis_out = torch.stack(vis_out.shape[0]*[
                        self.running_avg_weight_visual
                    ]).view(vis_out.shape + (1,) * dim_diff)
                
            elif caring_modality==1:
                dim_diff = len(visual.shape) - len(vis_out.shape)
                vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

                dim_diff = len(skeleton.shape) - len(sk_out.shape)
                sk_out = torch.stack(sk_out.shape[0]*[
                        self.running_avg_weight_skeleton
                    ]).view(sk_out.shape + (1,) * dim_diff)

        return visual * vis_out, skeleton * sk_out, scales, squeeze_array


def get_mmtm_outputs(eval_save_path, mmtm_recorded, key):
    with open(os.path.join(eval_save_path, 'history.pickle'), 'rb') as f:
        his_epo = pickle.load(f)
    
    print(his_epo.keys())
    data = []
    for batch in his_epo[key][0]:
        assert mmtm_recorded == len(batch)

        for mmtmid in range(len(batch)):
            if len(data)<mmtmid+1:
                data.append({})
            for i, viewdd in enumerate(batch[mmtmid]):
                data[mmtmid].setdefault('view_%d'%i, []).append(np.array(viewdd))
           
    for mmtmid in range(len(data)):
        for k, v in data[mmtmid].items():
            data[mmtmid][k] = np.concatenate(data[mmtmid][k])[np.argsort(his_epo['test_indices'][0])]  

    return data


def get_rescale_weights(eval_save_path, 
                        training_save_path, 
                        key='test_squeezedmaps_array_list',
                        validation=False, 
                        starting_mmtmindice = 1, 
                        mmtmpositions=4,
                        device=None,
                        ):
    data = get_mmtm_outputs(eval_save_path, mmtmpositions-starting_mmtmindice, key)
    
    with open(os.path.join(training_save_path, 'history.pickle'), 'rb') as f:
        his_ori = pickle.load(f)

    selected_indices = his_ori['val_indices'][0] if validation else his_ori['train_indices'][0] 
      
    mmtm_weights = []        
    for mmtmid in range(mmtmpositions):
        if mmtmid < starting_mmtmindice:
            mmtm_weights.append(None)
        else:
            weights = [data[mmtmid-starting_mmtmindice][k][selected_indices].mean(0) \
                        for k in sorted(data[mmtmid-starting_mmtmindice].keys())]
            if device is not None:
                weights = numpy_to_torch(weights)
                weights = torch_to(weights, device)
            mmtm_weights.append(weights)
        
    return mmtm_weights




