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
from Impedance_controller import ImpedencecontrolEnv

import franka_msgs.msg
import message_filters
import actionlib

from sensor_msgs.msg import JointState
from franka_gripper.msg import MoveGoal, MoveAction, GraspAction, GraspGoal
from control_msgs.msg import GripperCommandAction, GripperCommandGoal

FLAGS = flags.FLAGS
flags.DEFINE_string("robot_ip", None, "IP address of the robot.", required=True)
flags.DEFINE_string("load_gripper", 'false', "Whether or not to load the gripper.")

class Actions(gym.Env):
    def __init__(self):
        super(Actions, self).__init__()
        self.eepub = rospy.Publisher('/cartesian_impedance_controller/equilibrium_pose', geom_msg.PoseStamped, queue_size=10)
        self.client = Client("/cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node")
        rospy.Subscriber("/franka_state_controller/franka_states", franka_msgs.msg.FrankaState, self.franka_callback)
        rospy.Subscriber("/franka_state_controller/franka_states", franka_msgs.msg.FrankaState, self.GetEEforce)
    
    def set_reference_limitation(self):
        time.sleep(1)
        for direction in ['x', 'y', 'z', 'neg_x', 'neg_y', 'neg_z']:
            self.client.update_configuration({"translational_clip_" + direction: 0.005})
            self.client.update_configuration({"rotational_clip_" + direction: 0.04})
        time.sleep(1)
    
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

    def grasp_fail(self, gripper_width):
        # self.gripper_width = close_gripper()
        if gripper_width < 0.002:
            self.grippstate = 1
        else:
            self.grippstate = 0
        
        return self.grippstate

    def parm_to_selection(self, id):
        if id == 0:
            self.disassembly_state = self.move_up()

        elif id ==1:
            self.disassemblystate = self.move_down()

        elif id ==2:
            self.disassemblystate = self.move_left()

        elif id ==3:
            self.disassemblystate = self.move_right()

        elif id==4:
            self.disassemblystate = self.move_front()

        elif id==5:
            self.disassemblystate = self.move_back()

        return self.disassemblystate

    def move_up(self):
        EEP_x_ = self.F_T_EE[0, 3]
        EEP_y_ = self.F_T_EE[1, 3]
        EEP_z_ = self.F_T_EE[2, 3]
        Euler_angle_ =list(self.Euler_fl)
        # print(Euler_angle)
        self.quat_rr_ = tf.transformations.quaternion_from_euler(Euler_angle_[0], 
                                                    Euler_angle_[1], 
                                                    Euler_angle_[2]
                                                    )
        self.Position_ = np.array([EEP_x_, EEP_y_, EEP_z_]).astype(np.float32)
                
        self.currentOrn_ = self.quat_rr_
        self.Orientation_ = tf.transformations.euler_from_quaternion(self.currentOrn_)
                
                
        target = [self.Position_[0], self.Position_[1], self.Position_[2], 
                self.Orientation_[0], self.Orientation_[1], self.Orientation_[2]]
        fh = []
        while True:
            _ = ImpedencecontrolEnv.monitor_force_change(fh)
            stopf = _
            print("stop singal", stopf)
            if stopf == 0:
                # print("!!!!!!!is 0000000")
                self.ImpedencePosition(0, 0, 0.05, 0, 0, 0)
                # print("movemovemovemove")
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
                target0 = target + np.array([0, 0, 0.05, 0, 0, 0])
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

        disassemblydiff = abs(EEP_z_-EEP_z)
        if disassemblydiff >= 0.005:
            self.disassemblystate = 0
        else:
            self.disassemblystate = 1
        return self.disassemblystate

    def move_down(self):
        EEP_x_ = self.F_T_EE[0, 3]
        EEP_y_ = self.F_T_EE[1, 3]
        EEP_z_ = self.F_T_EE[2, 3]
        Euler_angle_ =list(self.Euler_fl)
        # print(Euler_angle)
        self.quat_rr_ = tf.transformations.quaternion_from_euler(Euler_angle_[0], 
                                                    Euler_angle_[1], 
                                                    Euler_angle_[2]
                                                    )
        self.Position_ = np.array([EEP_x_, EEP_y_, EEP_z_]).astype(np.float32)
                
        self.currentOrn_ = self.quat_rr_
        self.Orientation_ = tf.transformations.euler_from_quaternion(self.currentOrn_)
                
                
        target = [self.Position_[0], self.Position_[1], self.Position_[2], 
                self.Orientation_[0], self.Orientation_[1], self.Orientation_[2]]
        fh = []
        while True:
            _ = ImpedencecontrolEnv.monitor_force_change(fh)
            stopf = _
            print("stop singal", stopf)
            if stopf == 0:
                # print("!!!!!!!is 0000000")
                self.ImpedencePosition(0, 0, -0.05, 0, 0, 0)
                # print("movemovemovemove")
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
                target0 = target + np.array([0, 0, -0.05, 0, 0, 0])
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
        # EEP_z_ = self.F_T_EE[2, 3]
        # fh = []
        # while True:
        #     _ = ImpedencecontrolEnv.monitor_force_change(fh)
        #     stopf = _
        #     print("stop singal", stopf)
        #     if stopf == 0:
        #         self.ImpedencePosition(0, 0, -0.05, 0, 0, 0)
        #     if stopf == 1:
        #         self.ImpedencePosition(0,0,0,0,0,0)
        #         break
        disassemblydiff = abs(EEP_z_-EEP_z)
        if disassemblydiff >= 0.005:
            self.disassemblystate = 0
        else:
            self.disassemblystate = 1
        return self.disassemblystate
        

    def move_right(self):
        EEP_x_ = self.F_T_EE[0, 3]
        EEP_y_ = self.F_T_EE[1, 3]
        EEP_z_ = self.F_T_EE[2, 3]
        Euler_angle_ =list(self.Euler_fl)
        # print(Euler_angle)
        self.quat_rr_ = tf.transformations.quaternion_from_euler(Euler_angle_[0], 
                                                    Euler_angle_[1], 
                                                    Euler_angle_[2]
                                                    )
        self.Position_ = np.array([EEP_x_, EEP_y_, EEP_z_]).astype(np.float32)
                
        self.currentOrn_ = self.quat_rr_
        self.Orientation_ = tf.transformations.euler_from_quaternion(self.currentOrn_)
                
                
        target = [self.Position_[0], self.Position_[1], self.Position_[2], 
                self.Orientation_[0], self.Orientation_[1], self.Orientation_[2]]
        fh = []
        while True:
            _ = ImpedencecontrolEnv.monitor_force_change(fh)
            stopf = _
            print("stop singal", stopf)
            if stopf == 0:
                # print("!!!!!!!is 0000000")
                self.ImpedencePosition(0, 0.05, 0, 0, 0, 0)
                # print("movemovemovemove")
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
                target0 = target + np.array([0, 0.05, 0, 0, 0, 0])
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
        # EEP_y_ = self.F_T_EE[1, 3]
        # fh = []
        # while True:
        #     _ = ImpedencecontrolEnv.monitor_force_change(fh)
        #     stopf = _
        #     print("stop singal", stopf)
        #     if stopf == 0:
        #         self.ImpedencePosition(0, 0.05, 0, 0, 0, 0)
        #     if stopf == 1:
        #         self.ImpedencePosition(0,0,0,0,0,0)
        #         break
        disassemblydiff = abs(EEP_y_-EEP_y)
        if disassemblydiff >= 0.005:
            self.disassemblystate = 0
        else:
            self.disassemblystate = 1
        return self.disassemblystate
        

    def move_left(self):
        EEP_x_ = self.F_T_EE[0, 3]
        EEP_y_ = self.F_T_EE[1, 3]
        EEP_z_ = self.F_T_EE[2, 3]
        Euler_angle_ =list(self.Euler_fl)
        # print(Euler_angle)
        self.quat_rr_ = tf.transformations.quaternion_from_euler(Euler_angle_[0], 
                                                    Euler_angle_[1], 
                                                    Euler_angle_[2]
                                                    )
        self.Position_ = np.array([EEP_x_, EEP_y_, EEP_z_]).astype(np.float32)
                
        self.currentOrn_ = self.quat_rr_
        self.Orientation_ = tf.transformations.euler_from_quaternion(self.currentOrn_)
                
                
        target = [self.Position_[0], self.Position_[1], self.Position_[2], 
                self.Orientation_[0], self.Orientation_[1], self.Orientation_[2]]
        fh = []
        while True:
            _ = ImpedencecontrolEnv.monitor_force_change(fh)
            stopf = _
            print("stop singal", stopf)
            if stopf == 0:
                # print("!!!!!!!is 0000000")
                self.ImpedencePosition(0, -0.05, 0, 0, 0, 0)
                # print("movemovemovemove")
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
                target0 = target + np.array([0, -0.05, 0, 0, 0, 0])
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
        # while True:
        #     _ = ImpedencecontrolEnv.monitor_force_change(fh)
        #     stopf = _
        #     print("stop singal", stopf)
        #     if stopf == 0:
        #         self.ImpedencePosition(0, -0.05, 0, 0, 0, 0)
        #     if stopf == 1:
        #         self.ImpedencePosition(0,0,0,0,0,0)
        #         break
        disassemblydiff = abs(EEP_y_-EEP_y)
        if disassemblydiff >= 0.005:
            self.disassemblystate = 0
        else:
            self.disassemblystate = 1
        return self.disassemblystate
        

    def move_front(self):
        EEP_x_ = self.F_T_EE[0, 3]
        EEP_y_ = self.F_T_EE[1, 3]
        EEP_z_ = self.F_T_EE[2, 3]
        Euler_angle_ =list(self.Euler_fl)
        # print(Euler_angle)
        self.quat_rr_ = tf.transformations.quaternion_from_euler(Euler_angle_[0], 
                                                    Euler_angle_[1], 
                                                    Euler_angle_[2]
                                                    )
        self.Position_ = np.array([EEP_x_, EEP_y_, EEP_z_]).astype(np.float32)
                
        self.currentOrn_ = self.quat_rr_
        self.Orientation_ = tf.transformations.euler_from_quaternion(self.currentOrn_)
                
                
        target = [self.Position_[0], self.Position_[1], self.Position_[2], 
                self.Orientation_[0], self.Orientation_[1], self.Orientation_[2]]
        fh = []
        while True:
            _ = ImpedencecontrolEnv.monitor_force_change(fh)
            stopf = _
            print("stop singal", stopf)
            if stopf == 0:
                # print("!!!!!!!is 0000000")
                self.ImpedencePosition(0.05, 0, 0, 0, 0, 0)
                # print("movemovemovemove")
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
                target0 = target + np.array([0.05, 0, 0, 0, 0, 0])
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

        disassemblydiff = abs(EEP_x_-EEP_x)
        if disassemblydiff >= 0.005:
            self.disassemblystate = 0
        else:
            self.disassemblystate = 1
        return self.disassemblystate
        

    def move_back(self):
        EEP_x_ = self.F_T_EE[0, 3]
        EEP_y_ = self.F_T_EE[1, 3]
        EEP_z_ = self.F_T_EE[2, 3]
        Euler_angle_ =list(self.Euler_fl)
        # print(Euler_angle)
        self.quat_rr_ = tf.transformations.quaternion_from_euler(Euler_angle_[0], 
                                                    Euler_angle_[1], 
                                                    Euler_angle_[2]
                                                    )
        self.Position_ = np.array([EEP_x_, EEP_y_, EEP_z_]).astype(np.float32)
                
        self.currentOrn_ = self.quat_rr_
        self.Orientation_ = tf.transformations.euler_from_quaternion(self.currentOrn_)
                
                
        target = [self.Position_[0], self.Position_[1], self.Position_[2], 
                self.Orientation_[0], self.Orientation_[1], self.Orientation_[2]]
        fh = []
        while True:
            _ = ImpedencecontrolEnv.monitor_force_change(fh)
            stopf = _
            print("stop singal", stopf)
            if stopf == 0:
                # print("!!!!!!!is 0000000")
                self.ImpedencePosition(-0.05, 0, 0, 0, 0, 0)
                # print("movemovemovemove")
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
                target0 = target + np.array([-0.05, 0, 0, 0, 0, 0])
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
        disassemblydiff = abs(EEP_x_-EEP_x)
        if disassemblydiff >= 0.005:
            self.disassemblystate = 0
        else:
            self.disassemblystate = 1
        return self.disassemblystate
        

    def robot_control_grasptarget(self,target):
        open_gripper()
        print("Gripper opened.")
        time.sleep(1)

        self.grasptarget = target
        self.pre_grasptarget = target + np.array([0,0,0.2,0,0,0])

        self.MovetoPoint(self.pre_grasptarget)
        time.sleep(1)

        
        fh = []
        while True:
            _ = self.monitor_force_change(fh)
            stopf = _
            print("stop singal", stopf)
            if stopf == 0:
                # print("!!!!!!!is 0000000")
                self.MovetoPoint(self.grasptarget)
                
                # print("movemovemovemove")
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
                diff = abs (target  - self.grasptarget)
                if(diff <= 0.003).all():
                    self.approachfail = 0
                    stopf == 1
                    break
                
                # elif stopf == 1:
                #     break
        
            if stopf == 1:
                print("stop singal is 1")
                self.ImpedencePosition(0,0,0,0,0,0)
                self.approachfail = 1
                break
        

        
        time.sleep(1)

        self.graspwidth = close_gripper()
        return self.approachfail, self.graspwidth


    def robot_control_place(self):
        self.grasptarget = np.array([0.499,0.284,0.043,np.pi, 0, 0])
        self.pre_grasptarget = self.grasptarget + np.array([0,0,0.3,0,0,0])

        self.MovetoPoint(self.pre_grasptarget)
        time.sleep(1)
        self.MovetoPoint(self.grasptarget)
        time.sleep(1)

        open_gripper()


    def initialrobot(self):
        fh = self.force_history
        while True:
            _ = self.monitor_force_change(fh)
            stopf = _
            print("stop singal", stopf)
            if stopf == 0:
                self.ImpedencePosition(0,-0.03,0,0,0,0)
            if stopf == 1:
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
        msg.pose.position = geom_msg.Point(0.591, -0.034, 0.3147)
        quat = R.from_euler('xyz', [np.pi, 0, np.pi/2]).as_quat()
        msg.pose.orientation = geom_msg.Quaternion(quat[0], quat[1], quat[2], quat[3])
        # input("\033[33m\nObserve the surroundings. Press enter to move the robot to the initial position.\033[0m")
        self.eepub.publish(msg)
        time.sleep(1)
        print("reset!!!")

def open_gripper():
    node_process = subprocess.Popen(shlex.split('rosrun franka_interactive_controllers libfranka_gripper_run 1'), stdout = subprocess.PIPE, stderr = subprocess.PIPE, text=True)
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
    # ImpedencecontrolEnv.gripper_state_callback()

    print("Gripper Opened")


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

    return data_dict.get("width", [])


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
                    
        imp_controller = Actions()
        imp_controller.client

    
        imp_controller.reset_arm()
        time.sleep(1)
        imp_controller.set_reference_limitation()

        time.sleep(1)
        # open_gripper()

        close_gripper()
        time.sleep(2)

        # force = imp_controller.franka_callback()
        # print(force)
        time.sleep(2)
        
        imp_controller.move_up()
        # imp_controller.move_down()
        # imp_controller.move_left()
        # imp_controller.move_right()
        # imp_controller.move_front()
        # imp_controller.move_back()
        time.sleep(2)
        # target = np.array([0.591, -0.034, 0.3147, np.pi, 0, np.pi/2])

        # imp_controller.MovetoPoint(target)
        # time.sleep(1)

        # target_ = np.array([0.3, 0, 0.48, np.pi, 0, np.pi/2])

        # imp_controller.MovetoPoint(target_)
        # time.sleep(1)

        # target_0 = np.array([0.61, -0.166, 0.15, np.pi, 0, np.pi/2 + np.pi/4])
        # imp_controller.MovetoPoint(target_0)
        # time.sleep(1)

        # imp_controller.ImpedencePosition(0, 0, 0.05, 0, 0, 0)
        # time.sleep(1)
        # imp_controller.initialrobot()

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
