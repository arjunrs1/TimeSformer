import pickle
import os
import json
import torch
import numpy as np

path_prefix = "/vision/asomaya1/ego_exo/ProficiencyEstimation/TimeSformer/outputs"

camera_views = ['ego', 'exo_all']
test_annotations_file = "/vision/asomaya1/ego_exo/data/demonstrator_arxiv24_test.json"
is_v2 = True if "24" in test_annotations_file else False
is_val = True if "val" in test_annotations_file else False

mode = "448pFull"
kinetics_pretrain_path = "_K400_pretrain"
vlp_pretrain_path = "_vlp_pretrain"
vlpv2_pretrain_path = "_vlpv2_pretrain"
how_to_100m_pretrain_path = "_kinetics_pretrain" #NOTE: Not a typo, original was labeled incorrectly as kinetics
#ONLY ONE OF THE BELOW PRE_TRAIN OPTIONS CAN BE TRUE!!
use_kinetics_pretrain = False
use_vlp_pretrain = False
use_vlpv2_pretrain = True  
use_how_to_100m_pretrain = False

preds_file_name = "preds"
if is_v2:
    mode += "_v2"
    preds_file_name += "_v2"
if is_val:
    preds_file_name = "val_" + preds_file_name

with open(test_annotations_file, 'r') as file:
    data = json.load(file)

num_exs = len(data)

video_ids = list()
ego_preds = list()
exo_preds = list()

for item in data:
    # Extract video id from one of the video paths
    video_path = item["video_paths"]["ego"]  # Using "ego" path as an example
    if is_v2:
        video_id = video_path.split('/')[2]
    else:
        video_id = video_path.split('/')[1]
    video_ids.append(video_id)

for view in camera_views:
    preds = []
    original_view = view
    if use_kinetics_pretrain:
        view += kinetics_pretrain_path
    elif use_vlp_pretrain:
        view += vlp_pretrain_path
    elif use_vlpv2_pretrain:
        view += vlpv2_pretrain_path
    elif use_how_to_100m_pretrain:
        view += how_to_100m_pretrain_path
    trunc_mode = "448pFull" if is_v2 else mode
    with open(os.path.join(path_prefix, trunc_mode, view, preds_file_name + ".pkl"), "rb") as f:
        data = pickle.load(f)
        pred = data[1]
        if len(pred) != num_exs:
            prediction_sets = []
            for i in range(4):
                prediction_sets.append(pred[i::4, :])
            for p in prediction_sets:
                preds.append(p)
        else:
            preds.append(pred)
    concat_preds = torch.stack(preds,dim=0)
    if original_view == 'ego':
        ego_preds = concat_preds
    elif original_view == 'exo_all':
        exo_preds = concat_preds

softmax_preds = torch.nn.functional.softmax(torch.concat((ego_preds, exo_preds),axis=0), dim=-1)
ensemble_preds = torch.mean(softmax_preds, dim=0)
pred_idxs = np.array(torch.argmax(ensemble_preds, dim=1))

pred_idxs = [int(idx) for idx in pred_idxs]
#ego_preds_formatted = [ego_preds.squeeze(0)[i].numpy().tolist() for i in range(ego_preds.squeeze(0).shape[0])]
#exo_preds_formatted = [exo_preds[:, i, :].numpy().tolist() for i in range(exo_preds.shape[1])]

result = {
    "videos": list(video_ids),
    "predictions": list(pred_idxs),
}

output_file_path = "model_predictions_challenge"
if is_v2:
    output_file_path +=  "_v2"
output_file_path += ".json"
with open(output_file_path, 'w') as file:
    json.dump(result, file)