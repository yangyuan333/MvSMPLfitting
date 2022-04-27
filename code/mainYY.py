# -*- coding: utf-8 -*-
'''
 @FileName    : main.py
 @EditTime    : 2021-09-19 21:46:57
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import sys
import os

import os.path as osp

import time
import torch
import numpy as np
from cmd_parser import parse_config
from utils.init_guess import init_guess_nonlinear
from init import init
from utils.init_guess import init_guess, load_init, fix_params, guess_init
from utils.non_linear_solver import non_linear_solver
from utils.utils import save_results
def main(**args):

    dataset_obj, setting = init(**args)

    dtype = setting['dtype']
    device = setting['device']

    start = time.time()

    results = {}
    s_last = None # the name of last sequence
    setting['seq_start'] = False # indicate the first frame of the sequence
    for idx, data in enumerate(dataset_obj):
        serial = data['serial']
        if serial != s_last:
            setting['seq_start'] = True
            s_last = serial
        else:
            setting['seq_start'] = False
        # filter out the view without annotaion
        keypoints = data['keypoints']
        views = 0
        extrinsics = []
        intrinsics = []
        keyps = []
        img_paths = []
        imgs = []
        cameras = []
        GT_contacts = []
        for v in range(len(keypoints)):
            if keypoints[v] is not None:
                extrinsics.append(setting['extrinsics'][v])
                intrinsics.append(setting['intrinsics'][v])
                cameras.append(setting['cameras'][v])
                keyps.append(keypoints[v])
                img_paths.append(data['img_path'][v])
                imgs.append(data['img'][v])
                GT_contacts.append(data['GT_contacts'][v])
                views += 1

        setting['views'] = views
        setting['extris'] = np.array(extrinsics)
        setting['intris'] = np.array(intrinsics)
        setting['camera'] = cameras
        data['img'] = imgs
        data['img_path'] = img_paths
        data['keypoints'] = keyps
        data['GT_contacts'] = GT_contacts
        print('Processing: {}'.format(data['img_path']))

        if setting['global_init_type'] == 'linear':
            init_guess(setting, data, use_torso=True, **args) ## 根据2djoints->3djoints 估计初始旋转和平移
        else:
            init_guess_nonlinear(setting, data, results, use_torso=True, **args)
        
        fix_params(setting, scale=setting['fixed_scale'], shape=setting['fixed_shape']) ## 设置第一步初始化的全局旋转平移，选择是否优化scale和shape
        # linear solve
        print("linear solve, to do...")
        # non-linear solve
        results = non_linear_solver(setting, data, **args)
        # save results
        save_results(setting, data, results, **args)
        
    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
    print('Processing the data took: {}'.format(time_msg))

def transProx2coco(js):
    idx = [0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]
    return js[idx]

if __name__ == "__main__":

    import json
    import glob
    import os
    for path in glob.glob(os.path.join(R'H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\keypoints\vicon_03301_01','*')):
        print(os.path.basename(path))
        data = json.load(open(path,'rb'))
        print(1)

    sys.argv = [
        "",
        "--config=cfg_files/fit_smpl_GTcontact.yaml",
        "--global_init_type=linear",
        "--use_hands=True",
        "--use_face=True",
        "--use_contact=True",
        "--use_sdf=True",
        "--use_foot_contact=False",
        "--pose_format=coco25",
        "--scene=H:\YangYuan\ProjectData\HumanObject\dataset\PROX\prox_quantiative_dataset\scenes\\vicon_final.obj",
        "--body_segments_dir=H:\\YangYuan\\Code\\phy_program\\MvSMPLfitting\\body_segments",
        "--output_folder=outputProx_GTcontact_s1_06",
        "--use_GT_contact=True"
        ]
    args = parse_config()
    main(**args)