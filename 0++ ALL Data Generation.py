# === Necessary Imports ===
import mujoco
from mujoco import viewer
import time
import time as pytime  # avoid conflict with mujoco.time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from contextlib import nullcontext
import os
import itertools

# === Part 1: Run MuJoCo Manual Demonstration and Save to CSV ===
xml_path = r"D:\PhD\0PhD-Implementation\0ALOHA-ALL\mobile_aloha_sim-master\aloha_mujoco\aloha\meshes_mujoco\aloha_v1.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

ctrl_ranges = np.array([
    [-3.14158, 3.14158], [0, 3.14158], [0, 3.14158], [-2, 1.67], [-1.5708, 1.5708], [-3.14158, 3.14158], [0, 0.0475], [0, 0.0475],
    [-3.14158, 3.14158], [0, 3.14158], [0, 3.14158], [-2, 1.67], [-1.5708, 1.5708], [-3.14158, 3.14158], [0, 0.0475], [0, 0.0475]
])

demonstration_data = []
save_path = "NewDataShafiq.csv"
with viewer.launch_passive(model, data) as v:
    print("MuJoCo viewer launched. Use GUI to manually control the robot.")
    print("Press Esc or close the window to exit.")
    while v.is_running():
        mujoco.mj_step(model, data)
        v.sync()
        pytime.sleep(model.opt.timestep)

        ctrl_clipped = np.clip(data.ctrl[:len(ctrl_ranges)], ctrl_ranges[:, 0], ctrl_ranges[:, 1])
        timestep_data = {
            'time': float(data.time),
            'qpos': data.qpos.tolist(),
            'qvel': data.qvel.tolist(),
            'ctrl': ctrl_clipped.tolist()
        }
        qpos_clean = data.qpos.tolist()
        qvel_clean = data.qvel.tolist()
        ctrl_clean = ctrl_clipped.tolist()

        print(f"[Time: {data.time:.4f}]")
        print(f"  qpos : {qpos_clean}")
        print(f"  qvel : {qvel_clean}")
        print(f"  ctrl : {ctrl_clean}")
        demonstration_data.append(timestep_data)

with open(save_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['time'] +
                    [f'qpos_{i}' for i in range(len(data.qpos))] +
                    [f'qvel_{i}' for i in range(len(data.qvel))] +
                    [f'ctrl_{i}' for i in range(ctrl_ranges.shape[0])])
    for step in demonstration_data:
        row = [step['time']] + step['qpos'] + step['qvel'] + step['ctrl']
        writer.writerow(row)