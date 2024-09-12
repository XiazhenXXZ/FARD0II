import sys
import rospy
import numpy as np
import shlex
import time
# from psutil import Popen
import geometry_msgs.msg as geom_msg
import time
import subprocess
# from subprocess import PIPE
from dynamic_reconfigure.client import Client
from absl import app, flags, logging
from scipy.spatial.transform import Rotation as R
import os
import gymnasium as gym
import math
import tf
import tf.transformations
import re
import json
import threading

import franka_msgs.msg
import message_filters
import actionlib

from sensor_msgs.msg import JointState
from franka_gripper.msg import MoveGoal, MoveAction, GraspAction, GraspGoal
from control_msgs.msg import GripperCommandAction, GripperCommandGoal

FLAGS = flags.FLAGS
flags.DEFINE_string("robot_ip", None, "IP address of the robot.", required=True)
flags.DEFINE_string("load_gripper", 'false', "Whether or not to load the gripper.")

class ImpedencecontrolEnv(gym.Env):
    def __init__(self):
        super(ImpedencecontrolEnv, self).__init__()
        self.eepub = rospy.Publisher('/cartesian_impedance_controller/equilibrium_pose', geom_msg.PoseStamped, queue_size=10)
        self.client = Client("/cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node")
        

        # Force
        self.franka_EE_trans = []
        self.franka_EE_quat = []
        self.F_T_EE = np.empty((4,4))
        self.K_F_ext_hat_K = []
        self.ext_force_ee = []
        self.ext_torque_ee = []
        self.quat_fl = []
        self.quat_rr = []
        self.franka_fault = None

        self.force_history = []
        self.stop_flag = 0
        self.time_window = 5
        # self.threshold = 0.1
        self.max_steps = 20

        # sub and pub
        # self.sub = rospy.Subscriber("/franka_state_controller/franka_states", franka_msgs.msg.FrankaState, self.convert_to_geometry_msg, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber("/franka_state_controller/franka_states", franka_msgs.msg.FrankaState, self.franka_callback)

        rospy.Subscriber("/franka_state_controller/franka_states", franka_msgs.msg.FrankaState, self.GetEEforce)
        
        # sub.registerCallback(self.gripper_state_callback)

        # ts = message_filters.TimeSynchronizer([franka_sub, eeforce_sub, gripper_sub], queue_size=10)
        # ts.registerCallback(self.GetEEforce)
   

    
    def set_reference_limitation(self):
        time.sleep(1)
        for direction in ['x', 'y', 'z', 'neg_x', 'neg_y', 'neg_z']:
            self.client.update_configuration({"translational_clip_" + direction: 0.005})
            self.client.update_configuration({"rotational_clip_" + direction: 0.04})
        time.sleep(1)
    
    def GetEEforce(self,FandT):
        # self.gripper_width = gripper.position[0]
        self.forceandtorque= np.array(FandT.K_F_ext_hat_K)
        self.Force = self.forceandtorque[0:3]
        self.Torque = self.forceandtorque[3:6]
        # self.Force = self.franka_callself.GetEEforce()back[0:3]
        # self.Torque = self.franka_callback[3:6]
        Fx = self.Force[0]
        Fy = self.Force[1]
        Fz = self.Force[2]
        self.resultant_force = math.sqrt(Fx**2 + Fy**2 + Fz**2)
               
        # print("force:", Fz,Fy,Fz)
        
        return self.resultant_force
    
    def calculate_average_force(self, force_history):
        
        return np.mean(force_history)
    
    def is_average_force_change_abnormal(self,prev_avg_force, curr_avg_force):
        force_change_rate = np.abs(curr_avg_force - prev_avg_force) / (np.abs(prev_avg_force) + 1e-6)

        return np.any(force_change_rate > 0.5)
    
    def monitor_force_change(self, force_history):
        
        current_forces = self.resultant_force
        force_history.append(current_forces)
        
        if len(force_history) > self.time_window:
            print(len(force_history))
            force_history.pop(0)
        

        if len(force_history) == self.time_window:
            previous_average_force = self.calculate_average_force(force_history[:self.time_window // 2])
            current_average_force = self.calculate_average_force(force_history[self.time_window // 2:])

            if self.is_average_force_change_abnormal(previous_average_force, current_average_force):
                self.stop_flag = 1
    

        return self.stop_flag
        


    def ImpedencePosition(self, dx, dy, dz, da, db, dc):
        # time.sleep(0.2)
        time.sleep(0.5)
        EEP_x = self.F_T_EE[0, 3]
        # print(EEP_x)
        EEP_y = self.F_T_EE[1, 3]
        EEP_z = self.F_T_EE[2, 3]
        Euler_angle =list(self.Euler_fl)
        print(Euler_angle)
        self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                    Euler_angle[1], 
                                                    Euler_angle[2]
                                                    )
        self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
        print(self.Position)
        self.currentOrn = self.quat_rr
        self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
        print(self.Orientation)
        
        target = [self.Position[0]+dx, self.Position[1]+dy, self.Position[2]+dz, 
                  self.Orientation[0]+da, self.Orientation[1]+db, self.Orientation[2]+dc]
        print(target)
        
        

        msg = geom_msg.PoseStamped()
        msg.header.frame_id = "0"# input("\033[33mPress enter to move the robot down to position. \033[0m")
        msg.header.stamp = rospy.Time.now()
        msg.pose.position = geom_msg.Point(target[0], target[1], target[2])
        quat = R.from_euler('xyz', [target[3], target[4], target[5]]).as_quat()
        msg.pose.orientation = geom_msg.Quaternion(quat[0], quat[1], quat[2], quat[3])
        self.eepub.publish(msg)
        time.sleep(1)


    def initialrobot(self):
        fh = self.force_history
        while True:
            _ = self.monitor_force_change(fh)
            stopf = _
            print("stop singal", stopf)
            if stopf == 0:
                print("!!!!!!!is 0000000")
                target0 = np.array([0.61, -0.256, 0.2, np.pi, 0, np.pi/2 + np.pi/4])
                self.MovetoPoint(target0)
                print("movemovemovemove")
                EEP_x = self.F_T_EE[0, 3]
                EEP_y = self.F_T_EE[1, 3]
                EEP_z = self.F_T_EE[2, 3]
                Euler_angle =list(self.Euler_fl)
                print(Euler_angle)
                self.quat_rr = tf.transformations.quaternion_from_euler(Euler_angle[0], 
                                                            Euler_angle[1], 
                                                            Euler_angle[2]
                                                            )
                self.Position = np.array([EEP_x, EEP_y, EEP_z]).astype(np.float32)
                
                self.currentOrn = self.quat_rr
                self.Orientation = tf.transformations.euler_from_quaternion(self.currentOrn)
                
                
                target = [self.Position[0], self.Position[1], self.Position[2], 
                        self.Orientation[0], self.Orientation[1], self.Orientation[2]]
                diff = abs (target  - target0)
                if(diff <= 0.003).all():
                    stopf == 1
                    break
                
                # elif stopf == 1:
                #     break
        
            if stopf == 1:
                print("stop singal is 1")
                self.ImpedencePosition(0,0,0,0,0,0)
                break
    
    def MovetoPoint(self, Target):
        time.sleep(1)
        target = Target
        msg = geom_msg.PoseStamped()
        msg.header.frame_id = "0"# input("\033[33mPress enter to move the robot down to position. \033[0m")
        msg.header.stamp = rospy.Time.now()
        msg.pose.position = geom_msg.Point(target[0], target[1], target[2])
        quat = R.from_euler('xyz', [target[3], target[4], target[5]]).as_quat()
        msg.pose.orientation = geom_msg.Quaternion(quat[0], quat[1], quat[2], quat[3])
        self.eepub.publish(msg)
        time.sleep(1)
        print("success!!!!!!!!!!!!!!!!!!!!!!")

    def franka_callback(self, data):
        # print(data)
        self.K_F_ext_hat_K = np.array(data.K_F_ext_hat_K)
        # print(self.K_F_ext_hat_K)

        # Tip of finger gripper
        self.O_T_EE = np.array(data.O_T_EE).reshape(4, 4).T
        # print(O_T_EE[:3, :3])
        quat_ee = tf.transformations.quaternion_from_matrix(self.O_T_EE)        

        # Flange of robot
        self.F_T_EE_ = np.array(data.F_T_EE).reshape(4, 4).T
        self.F_T_EE_1 = np.asmatrix(self.O_T_EE) * np.linalg.inv(np.asmatrix(self.F_T_EE_))

        # Hand TCP of robot
        self.hand_TCP = np.array([[0.7071, 0.7071, 0, 0],
                                 [-0.7071, 0.7071, 0, 0],
                                 [0, 0, 1, 0.1034],
                                 [0, 0, 0, 1]])
        self.F_T_EE = np.asmatrix(self.F_T_EE_1) * np.asmatrix(self.hand_TCP)

        self.quat_fl_ = tf.transformations.quaternion_from_matrix(self.F_T_EE_1)
        # print(quat_fl)
        self.Euler_fl_ = tf.transformations.euler_from_quaternion(self.quat_fl_)

        # print("self.F_T_EE:", self.F_T_EE)
        self.quat_fl = tf.transformations.quaternion_from_matrix(self.F_T_EE)
        # print(quat_fl)
        self.Euler_fl = tf.transformations.euler_from_quaternion(self.quat_fl)

        return self.K_F_ext_hat_K, self.F_T_EE
        # print("Force:", self.K_F_ext_hat_K)
        # self.ext_force_ee = self.K_F_ext_hat_K[0:2]
        # self.ext_torque_ee = self.K_F_ext_hat_K[3:5]
    

    
    def franka_state_callback(self, msg):
        self.cart_pose_trans_mat = np.asarray(msg.O_T_EE).reshape(4,4,order='F')
        self.cartesian_pose = {
            'position': self.cart_pose_trans_mat[:3,3],
            'orientation': tf.transformations.quaternion_from_matrix(self.cart_pose_trans_mat[:3,:3]) }
        self.franka_fault = self.franka_fault or msg.last_motion_errors.joint_position_limits_violation or msg.last_motion_errors.cartesian_reflex
 

    def reset_arm(self):
        time.sleep(1)
        msg = geom_msg.PoseStamped()
        msg.header.frame_id = "0"
        msg.header.stamp = rospy.Time.now()
        msg.pose.position = geom_msg.Point(0.55, 0, 0.35)
        quat = R.from_euler('xyz', [np.pi, 0, 0]).as_quat()
        msg.pose.orientation = geom_msg.Quaternion(quat[0], quat[1], quat[2], quat[3])
        # input("\033[33m\nObserve the surroundings. Press enter to move the robot to the initial position.\033[0m")
        self.eepub.publish(msg)
        time.sleep(1)
        print("reset!!!")

    # def gripper_state_callback(self, gri):
    #     # Extract the gripper width from the JointState message
    #     self.gripper_width = gri.position[0]  # Assuming the first position value represents the gripper width
    #     print(self.gripper_width)

    #     return self.gripper_width
    
    def get_gripper_state(self):
        return self.gripper_width

def gripper_callback(msg):
    print(msg)
    width = msg.position[0]
    print(width)
    print("gripper!!!!")

    return width

def gripper_listener():
    # # Initialize the ROS node
    # rospy.init_node('gripper_width_listener', anonymous=True)

    # # Subscribe to the gripper state topic
    # rospy.Subscriber("/franka_gripper/joint_states", JointState, gripper_callback)

    # # Keep the node running
    # # rospy.spin()
    print("listener")

def open_gripper():
    node_process = subprocess.Popen(shlex.split('rosrun franka_interactive_controllers libfranka_gripper_run 1'), stdout = subprocess.PIPE, stderr = subprocess.PIPE, text=True)
    # print("111111")
    stdout, stderr = node_process.communicate()
    message = stdout
    msg = message.strip()
    # print(msg)        # target_0 = np.array([0.61, -0.166, 0.15, np.pi, 0, np.pi/2 + np.pi/4])
        # imp_controller.MovetoPoint(target_0)
        # time.sleep(1)
    match = re.search(r'\{.*\}', msg)
    # print("2222222")
    if match:
        # print(match)
        try:
            dict_str = match.group(0)
            # print(dict_str)
            data_dict = json.loads(dict_str)
            # print(data_dict)
        except json.JSONDecodeError as e:
            print(e)

    else:
        print("none!!!!!!!!!!")

    if stderr:
        print("stderr:", stderr)
    print("msg!!!!!!!!!!!!!!!!!!:", data_dict.get("width", []))
    gripper_width = data_dict.get("width", [])
    node_process.wait()  # Wait for the command to complete
    # ImpedencecontrolEnv.gripper_state_callback()

    print("Gripper Opened")
    return gripper_width


def close_gripper():
    node_process = subprocess.Popen(shlex.split('rosrun franka_interactive_controllers libfranka_gripper_run 0'),stdout = subprocess.PIPE, stderr = subprocess.PIPE, text=True)
    # print("111111")
    stdout, stderr = node_process.communicate()
    message = stdout
    msg = message.strip()
    # print(msg)
    match = re.search(r'\{.*\}', msg)
    # print("2222222")
    if match:
        # print(match)
        try:
            dict_str = match.group(0)
            # print(dict_str)
            data_dict = json.loads(dict_str)
            # print(data_dict)
        except json.JSONDecodeError as e:
            print(e)

    else:
        print("none!!!!!!!!!!")

    if stderr:
        print("stderr:", stderr)
    print("msg!!!!!!!!!!!!!!!!!!:", data_dict.get("width", []))
    node_process.wait()  # Wait for the command to complete
    print("Gripper Closed")


def activate(FLAGS):
    # roscore = subprocess.Popen('roscore')
    time.sleep(1)

    impedence_controller = subprocess.Popen(['roslaunch', 'serl_franka_controllers', 'impedance.launch',
                                            f'robot_ip:={FLAGS.robot_ip}', f'load_gripper:={FLAGS.load_gripper}'],
                                            stdout=subprocess.PIPE)
    time.sleep(1)
    rospy.init_node('franka_control_api')

    return impedence_controller

def terminate(impedence_controller):
    impedence_controller.terminate()
    # roscore.terminate()


def main(_):
    try:
        roscore = subprocess.Popen('roscore')
        time.sleep(1)

        impedence_controller = subprocess.Popen(['roslaunch', 'serl_franka_controllers', 'impedance.launch',
                                                f'robot_ip:={FLAGS.robot_ip}', f'load_gripper:={FLAGS.load_gripper}'],
                                                stdout=subprocess.PIPE)
        time.sleep(1)
        rospy.init_node('franka_control_api')
                    
        imp_controller = ImpedencecontrolEnv()
        imp_controller.client

    
        imp_controller.reset_arm()
        time.sleep(1)
        imp_controller.set_reference_limitation()
        # roscore.terminate()

        time.sleep(1)
        # rospy.spin()
        
        # open_gripper()

        close_gripper()
        # grasp_client(0.04)
        time.sleep(2)
        # gripper_listener()
        # gripper_callback()
        # time.sleep(2)
        
        
        # force = imp_controller.franka_callback()
        # print(force)
        time.sleep(2)
        
        target = np.array([0.591, -0.034, 0.3147, np.pi, 0, np.pi/2])

        imp_controller.MovetoPoint(target)
        time.sleep(1)

        target_ = np.array([0.3, 0, 0.28, np.pi, 0, np.pi/2])

        imp_controller.MovetoPoint(target_)
        time.sleep(1)

        # target_0 = np.array([0.61, -0.166, 0.15, np.pi, 0, np.pi/2 + np.pi/4])
        # imp_controller.MovetoPoint(target_0)
        # time.sleep(1)

        imp_controller.ImpedencePosition(0, 0, 0.05, 0, 0, 0)
        time.sleep(1)


        imp_controller.initialrobot()
        time.sleep(0.5)

        imp_controller.ImpedencePosition(0, 0, 0.03, 0, 0, 0)

        


        impedence_controller.terminate()
        roscore.terminate()
        sys.exit()

    except:
        rospy.logerr("Error occured. Terminating the controller.")
        impedence_controller.terminate()
        roscore.terminate()
        sys.exit()


if __name__ == "__main__":
    app.run(main)
    #!/usr/bin/env python

# import rospy
# from franka_gripper.srv import GraspEpsilon  # Example service, replace with the actual service type

# def get_gripper_width():
#     rospy.wait_for_service('/franka_gripper/get_width')
#     try:
#         get_width = rospy.ServiceProxy('/franka_gripper/get_width', GraspEpsilon)  # Replace with the actual service type
#         response = get_width()
#         rospy.loginfo(f"Current gripper width: {response.width}")
#     except rospy.ServiceException as e:
#         rospy.logerr(f"Service call failed: {e}")

# if __name__ == '__main__':
#     rospy.init_node('get_gripper_width_node')
#     get_gripper_width()
