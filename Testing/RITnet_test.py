#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:37:59 2019

@author: aaa
"""
import torch
from dataset import IrisDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from dataset import transform
import os
from opt import parse_args
from models import model_dict
from tqdm import tqdm
from utils import get_predictions

# %%

if __name__ == '__main__':

    args = parse_args()

    device = torch.device("cpu")

    model = model_dict["densenet"]
    model = model.to(device)
    filename = "C:/Users/Caleb/OneDrive/BYUI/Fall2024/SeniorProject/GithubProject/Models/RITnet.pkl"

    if not os.path.exists(filename):
        print("model path not found !!!")
        exit(1)

    model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()

    test_set = IrisDataset(filepath='Dataset/',
                           split='test', transform=transform)

    testloader = DataLoader(test_set, batch_size=5,
                            shuffle=False, num_workers=0)

    counter = 0

    os.makedirs('RITtest/', exist_ok=True)
    os.makedirs('RITtest/labels/', exist_ok=True)
    os.makedirs('RITtest/output/', exist_ok=True)
    os.makedirs('RITtest/mask/', exist_ok=True)

    with torch.no_grad():
        for i, batchdata in tqdm(enumerate(testloader), total=len(testloader)):
            img, labels, index, x, y = batchdata
            data = img.to(device)
            output = model(data)
            predict = get_predictions(output)
            for j in range(len(index)):
                np.save('RITtest/labels/{}.npy'.format(index[j]), predict[j].cpu().numpy())
                try:
                    plt.imsave('RITtest/output/{}.jpg'.format(index[j]), 255 * labels[j].cpu().numpy())
                except:
                    pass

                pred_img = predict[j].cpu().numpy() / 3.0
                inp = img[j].squeeze() * 0.5 + 0.5
                img_orig = np.clip(inp, 0, 1)
                img_orig = np.array(img_orig)
                combine = np.hstack([img_orig, pred_img])
                plt.imsave('RITtest/mask/{}.jpg'.format(index[j]), combine)

