import os
import sys
import ctypes
import time
import select
import termios
import tty
import numpy as np
import cv2
import csv
from datetime import datetime

# --- Haptics Imports ---
from pyOpenHaptics.hd_device import HapticDevice
import pyOpenHaptics.hd as hd
from pyOpenHaptics.hd_callback import hd_callback
from dataclasses import dataclass, field

# --- Mujoco Imports ---
import mujoco
from mujoco import viewer

# =============================================================================
# 1. HAPTIC DEVICE SETUP
# =============================================================================

os.environ['GTDD_HOME'] = os.path.expanduser('~/.3dsystems')

def find_haptics_library():
    likely_path = '/home/shafiq/haptics_ws/drivers/openhaptics_3.4-0-developer-edition-amd64/usr/lib'
    if os.path.exists(os.path.join(likely_path, 'libHD.so')):
        return likely_path

    system_libs = ['/usr/lib', '/usr/local/lib', '/usr/lib/x86_64-linux-gnu']
    for path in system_libs:
        if os.path.exists(os.path.join(path, 'libHD.so')):
            return path

    search_root = '/home/shafiq/haptics_ws'
    for root, dirs, files in os.walk(search_root):
        if 'libHD.so' in files:
            return root

    return None

lib_path = find_haptics_library()

if not lib_path:
    print("✗ ERROR: Could not find libHD.so.")
    sys.exit(1)

ctypes.CDLL(os.path.join(lib_path, 'libHD.so'))
os.environ['LD_LIBRARY_PATH'] = lib_path
print("✓ Haptic library loaded")

@dataclass
class DeviceState:
    full_joints: list = field(default_factory=lambda: [0]*6)
    btn_top: bool = False
    btn_bottom: bool = False

device_state = DeviceState()

@hd_callback
def state_callback():
    global device_state
    try:
        motors = hd.get_joints()
        gimbals = hd.get_gimbals()

        device_state.full_joints = [
            motors[0], motors[1], motors[2],
            gimbals[0], gimbals[1], gimbals[2]
        ]

        btn_mask = hd.get_buttons()
        device_state.btn_top = (btn_mask & 1) != 0
        device_state.btn_bottom = (btn_mask & 2) != 0

        hd.set_force([0, 0, 0])

    except:
        pass

# =============================================================================
# 2. MUJOCO SETUP
# =============================================================================

XML_PATH = "/home/shafiq/Desktop/0ALOHA-ALL/mobile_aloha_sim-master/aloha_mujoco/aloha/meshes_mujoco/aloha_v1.xml"

def clamp(v, mn, mx):
    return np.clip(v, mn, mx)

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
print("✓ Mujoco model loaded")

ctrl_ranges = np.array([
    [-3.14158, 3.14158], [0, 3.14158], [0, 3.14158], [-2, 1.67],
    [-1.5708, 1.5708], [-3.14158, 3.14158], [0, 0.0475], [0, 0.0475],
    [-3.14158, 3.14158], [0, 3.14158], [0, 3.14158], [-2, 1.67],
    [-1.5708, 1.5708], [-3.14158, 3.14158], [0, 0.0475], [0, 0.0475]
])

# Actuator indices
right_arm_joint_names = [f"fr_joint{i}" for i in range(1, 9)]
right_arm_actuator_ctrl_indices = [model.actuator(name).id for name in right_arm_joint_names]
right_gripper_indices = [
    model.actuator("fr_joint7").id,
    model.actuator("fr_joint8").id
]

left_arm_joint_names = [f"fl_joint{i}" for i in range(1, 9)]
left_arm_actuator_ctrl_indices = [model.actuator(name).id for name in left_arm_joint_names]
left_gripper_indices = [
    model.actuator("fl_joint7").id,
    model.actuator("fl_joint8").id
]

right_arm_state = np.zeros(8)
left_arm_state = np.zeros(8)
right_gripper_val = 0.01

GRIPPER_MIN = ctrl_ranges[14][0]
GRIPPER_MAX = ctrl_ranges[14][1]
GRIPPER_STEP = 0.001

# =============================================================================
# 3. MAIN LOOP + CSV LOGGING
# =============================================================================

if __name__ == "__main__":

    demonstration_data = []
    save_path = f"NewDataShafiq_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    old_settings = termios.tcgetattr(sys.stdin)

    haptic_device = HapticDevice(callback=state_callback, scheduler_type="async")
    time.sleep(0.2)
    print("✓ Haptic connected")

    try:
        tty.setcbreak(sys.stdin.fileno())

        with viewer.launch_passive(model, data) as v:
            print("Simulation running. Press ESC to exit.")

            running = True
            while running:

                if select.select([sys.stdin], [], [], 0)[0]:
                    if sys.stdin.read(1) == '\x1b':
                        running = False

                h_joints = device_state.full_joints
                h_btn_top = device_state.btn_top
                h_btn_bottom = device_state.btn_bottom

                # Map haptic to right arm
                for i in range(6):
                    right_arm_state[i] = clamp(
                        h_joints[i],
                        ctrl_ranges[i][0],
                        ctrl_ranges[i][1]
                    )

                # Gripper logic
                if h_btn_top:
                    right_gripper_val -= GRIPPER_STEP
                elif h_btn_bottom:
                    right_gripper_val += GRIPPER_STEP

                right_gripper_val = clamp(
                    right_gripper_val,
                    GRIPPER_MIN,
                    GRIPPER_MAX
                )

                right_arm_state[6] = right_gripper_val
                right_arm_state[7] = right_gripper_val

                # Apply control
                for i in range(6):
                    data.ctrl[right_arm_actuator_ctrl_indices[i]] = right_arm_state[i]

                data.ctrl[right_gripper_indices[0]] = right_arm_state[6]
                data.ctrl[right_gripper_indices[1]] = right_arm_state[7]

                # Hold left arm fixed
                for i in range(6):
                    data.ctrl[left_arm_actuator_ctrl_indices[i]] = left_arm_state[i]

                data.ctrl[left_gripper_indices[0]] = left_arm_state[6]
                data.ctrl[left_gripper_indices[1]] = left_arm_state[7]

                # ==============================
                # LOG DATA (Manual Demo Pattern)
                # ==============================
                ctrl_clipped = np.clip(
                    data.ctrl[:len(ctrl_ranges)],
                    ctrl_ranges[:, 0],
                    ctrl_ranges[:, 1]
                )

                timestep_data = {
                    'time': float(data.time),
                    'qpos': data.qpos.tolist(),
                    'qvel': data.qvel.tolist(),
                    'ctrl': ctrl_clipped.tolist()
                }

                demonstration_data.append(timestep_data)

                mujoco.mj_step(model, data)
                v.sync()
                time.sleep(model.opt.timestep)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        haptic_device.close()

        # ==============================
        # SAVE CSV (Same Format)
        # ==============================
        try:
            with open(save_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                writer.writerow(
                    ['time'] +
                    [f'qpos_{i}' for i in range(len(data.qpos))] +
                    [f'qvel_{i}' for i in range(len(data.qvel))] +
                    [f'ctrl_{i}' for i in range(ctrl_ranges.shape[0])]
                )

                for step in demonstration_data:
                    row = (
                        [step['time']] +
                        step['qpos'] +
                        step['qvel'] +
                        step['ctrl']
                    )
                    writer.writerow(row)

            print(f"\n✓ Simulation data saved to: {save_path}")

        except Exception as e:
            print(f"✗ CSV Save Error: {e}")

        print("Device Closed safely.")
