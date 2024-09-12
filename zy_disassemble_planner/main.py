import random
import time
import numpy as np
import tf
import os
import yaml
import threading

from init_results_detection import inspector, init_generation, ObjectDetector
from Actions import Actions, open_gripper, close_gripper


# place
def where2go(object_index):
    Destination = np.array([0.577,-0.347,0.125], dtype=np.float32)

    return Destination
        
# Function to convert a list to a YAML file
def list_to_yaml(list_data, file_name):
    # Create a dictionary with the list data under 'Policy' -> 'Action'
    data = {'Policy': {'Action': list_data}}
    
    # Open the specified file in write mode and dump the dictionary as YAML
    with open(file_name, 'w') as yamlfile:
        yaml.dump(data, yamlfile, default_flow_style=False)

# Define the file path where the YAML file will be saved
file_path = '/home/yuezang/catkin_ws/src/franka_real_demo/src/franka_real_demo/scripts/zy_disassemble_planner/config.yaml'

def main():
    policy = []
    Object_init = []  # random choice use

    init_id, init_condition = init_generation()

    for id_info in init_id:
        Class_index = id_info.get("id", [])
        Object_init.append(Class_index)
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

    for _ in range(10):
        id_select = random.choice(Object_init)
        print("id:", id_select)
        action_list = [0, 1, 2, 3, 4, 5]
        camera_thread = threading.Thread(target=inspector(), daemon=True)
        camera_thread.start()
        for obj in init_condition:
            Object_info = obj.get(str(id_select), None)
            if Object_info is None:
                continue
            print(Object_info)
            approach_pos_ = Object_info.get("Detected_object_pose", [])
            place_pos_ = Object_info.get("Destination", [])
            print(approach_pos_)
            print(place_pos_)
            open_gripper()
            Approachfail, grasp_width = Actions.robot_control_grasptarget(approach_pos_)
            if Approachfail == 1:
                execute_fail_approach = 1
            else:
                execute_fail_approach = 0
            
            if execute_fail_approach == 1:
                Object_init.remove(id_select)
                break

            else:
                grasp_fail = Actions.grasp_fail(gripper_width=grasp_width)
                if grasp_fail == 1:
                    break
                else:
                    action_select = np.random.randint(0, len(action_list))
                    print("action:", action_select)
                    disassembly_state = Actions.parm_to_selection(action_select)
                    if disassembly_state == 1:
                        action_list.remove(action_select)
                        action_select_ = np.random.randint(0, len(action_list))
                        execute_action_ = execute_action(action_select_, approach_pos_, place_pos_)
                        execute_action = execute_action_
                        if len(action_list) == 0:
                            execute_action = False
                            Object_init.remove(id_select)
                            break
                        else:
                            pass
                    
                    else:
                        Actions.robot_control_place(place_pos_)
                        result = ObjectDetector.detect_in_place()
                        if result == True:
                            print("success")
                            for obj in init_condition:
                                Object_info = obj.get(str(action_select), None)
                                if Object_info is not None:
                                    Object_info["Actions"] = action_select  

                            print("Updated init_from_scan for success action:")
                            for obj in init_condition:
                                print(obj)
                                
                            policy_step = {
                                'id': id_select,
                                'actions': action_select,
                                'destination': place_pos_
                            }
                            policy.append(policy_step)
                            break
                            # Object_init.remove(action_select)

                        else:
                            Object_init.remove(id_select)
                        
                    
                        for policy_step in policy:
                            print(policy_step)

    # Convert the list to a YAML file and save it to the specified path
    list_to_yaml(policy, file_path)

    # Print the path of the created configuration file
    print(f"Configuration file created: {file_path}")

if __name__ == '__main__':
    main()