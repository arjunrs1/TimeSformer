import pickle
import os
import json
import torch
import numpy as np

camera_views = ['ego', 'exo_all']
test_annotations_file = "/vision/asomaya1/ego_exo/data/demonstrator_arxiv23_test.json"

mode = "448pFull"
kinetics_pretrain_path = "_kinetics_pretrain"
vlp_pretrain_path = "_vlp_pretrain"
vlpv2_pretrain_path = "_vlpv2_pretrain"
#ONLY ONE OF THE BELOW PRE_TRAIN OPTIONS CAN BE TRUE!!
use_kinetics_pretrain = False
use_vlp_pretrain = True
use_vlpv2_pretrain = False  
multi_take_eval = False

with open(test_annotations_file, 'r') as file:
    data = json.load(file)

num_exs = len(data)

video_ids = list()
ego_preds = list()
exo_preds = list()

for item in data:
    # Extract video id from one of the video paths
    video_path = item["video_paths"]["ego"]  # Using "ego" path as an example
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
    with open(os.path.join(mode, view, "preds.pkl"), "rb") as f:
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

ego_preds_formatted = [ego_preds.squeeze(0)[i].numpy().tolist() for i in range(ego_preds.squeeze(0).shape[0])]
exo_preds_formatted = [exo_preds[:, i, :].numpy().tolist() for i in range(exo_preds.shape[1])]

result = {
    "videos": list(video_ids),
    "ego_model_predictions": list(ego_preds_formatted),
    "exo_model_predictions": list(exo_preds_formatted)
}

output_file_path = "model_predictions.json"
with open(output_file_path, 'w') as file:
    json.dump(result, file)