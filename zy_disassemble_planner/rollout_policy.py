import random
import time
import numpy as np
import tf
import os
import yaml

from object_detection2init import init_generation
from Actions import Actions

def execute_action(id, pick_pos, place_pos):
    Actions.arm_initial_conf()
    Actions.open_franka_gripper()
    # print("pick_pose in execute action:", pick_pos[1])
    Actions.visual_grasp_policy(id, pick_pos[0], pick_pos[1])
    print("place_pose in execute action:", place_pos)
    Actions.place(place_pos)

def where2go(object_index):
    if object_index == float(27.0):
        Destination = np.array([0.608, -0.1198, 0], dtype=np.float32)
    else:
        Destination = np.array([0.625, -0.298, 0], dtype=np.float32)

    return Destination

def read_yaml(file_name):
    with open(file_name, 'r') as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)

        items = data['Policy']['Action']
        
    return items

yaml_file = '/home/yuezang/catkin_ws/src/franka_real_demo/src/franka_real_demo/scripts/TASK_Motion_planner/config.yaml'
list_from_yaml = read_yaml(yaml_file)

print("list_from_yaml:", list_from_yaml)

for item in list_from_yaml:
    print("Processing item:", item)

Policy_list = []
for item in list_from_yaml:
    Policy_list.append(str(item)) 

print("Policy_list:", Policy_list)

Object_init = []  
Object_init_str = []
init_id, init_condition = init_generation()

for id_info in init_id:
    Class_index = id_info.get("id", [])
    Object_init.append(Class_index)
    Object_init_str.append(str(Class_index))
    print("id:", Object_init)

for _ in range(len(Object_init)):
    for obj in init_condition:
        # Access the nested dictionary using Object_init[_] as key
        Object_info = obj.get(str(Object_init[_]), None)
        if Object_info is not None:
            # print("Object_info:", Object_info)
            Destination = where2go(float(Object_init[_]))
            # print("Destination:", Destination)
            Object_info["Destination"] = Destination
            # print("Updated Object_info:", Object_info)

print("Updated init_condition:")
for obj in init_condition:
    print(obj)
print(Policy_list)
print(Object_init)
if sorted(Policy_list) == sorted(Object_init_str):
    print("pass!!!!!!!!!!!!!!!!")
    for action in Policy_list:
        for obj in init_condition:
                Object_info = obj.get(str(action), None)
                if Object_info is None:
                    continue
                print(Object_info)
                pick_pos_ = Object_info.get("Detected_object_information", [])
                place_pos_ = Object_info.get("Destination", [])
                print(pick_pos_)
                print(place_pos_)

                execute_action(action, pick_pos_, place_pos_)
else:
    print("problem!!!!!!!!!!!!!!")

