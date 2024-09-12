import copy
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

from pca_orientation import PCACalculator
from transformation import CameraPoseCalculator
from width_calculation import WidthComputer


class ObjectDetector:
    def __init__(self, model_directory, w=640, h=480):
        self.model = YOLO(model_directory)
        # self.target_class_index = target_class_index
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
        self.align = rs.align(rs.stream.color)
        self.pipeline.start(self.config)

    def detect_objects_and_depth(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return [], None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

        results = self.model(color_image, conf=0.1)
        objects_info = []

        for r in results:
            boxes = r.boxes
            # print("##########boxes#############:", boxes)
            masks = r.masks
            for index in range(len(boxes)):
                b = boxes[index].xyxy[0].to('cpu').detach().numpy().copy()
                c = boxes[index].cls

                cv2.rectangle(depth_colormap, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), thickness=2,
                              lineType=cv2.LINE_4)
                cv2.putText(depth_colormap, text=self.model.names[int(c)], org=(int(b[0]), int(b[1])),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 255), thickness=2,
                            lineType=cv2.LINE_4)

                center_x = int((b[0] + b[2]) / 2)
                center_y = int((b[1] + b[3]) / 2)
                depth = depth_frame.get_distance(center_x, center_y)
                # print(depth)

                # Draw the center point with depth info
                cv2.circle(depth_colormap, (center_x, center_y), radius=5, color=(255, 0, 0), thickness=-1)
                depth_text = f"Depth: {depth:.2f} meters"
                cv2.putText(depth_colormap, depth_text, (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0), 2)

                boundary_depth = []
                for boundary_point in masks[index].xy[0]:  # mask[0] contains the boundary points
                    boundary_x, boundary_y = int(boundary_point[0]), int(boundary_point[1])
                    point_depth = depth_frame.get_distance(boundary_x, boundary_y)
                    boundary_depth.append(point_depth)

                objects_info.append({
                    "class": self.model.names[int(c)],
                    "class_index": c,
                    "box_coordinates": b,
                    "object_center": (center_x, center_y),
                    "center_depth": depth,
                    "boundary_mask": masks[index].xy, #if int(c) == self.target_class_index else None,
                    "boundary_depth": boundary_depth
                })

                annotated_frame = results[0].plot()
                cv2.imshow("color_image", annotated_frame)
                cv2.imshow("depth_image", depth_colormap)

    def detect_in_place(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return [], None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        x, y, w, h = 140, 300, 280, 180
        cropped_color_image = color_image[y:y+h, x:x+w]
        cropped_depth_image = depth_image[y:y+h, x:x+w]
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(cropped_depth_image, alpha=0.08), cv2.COLORMAP_JET)

        results = self.model(cropped_color_image, conf=0.1)
        objects_info = []

        for r in results:
            boxes = r.boxes
            # print("##########boxes#############:", boxes)
            masks = r.masks
            for index in range(len(boxes)):
                b = boxes[index].xyxy[0].to('cpu').detach().numpy().copy()
                c = boxes[index].cls

                cv2.rectangle(depth_colormap, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), thickness=2,
                              lineType=cv2.LINE_4)
                cv2.putText(depth_colormap, text=self.model.names[int(c)], org=(int(b[0]), int(b[1])),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 255), thickness=2,
                            lineType=cv2.LINE_4)

                center_x = int((b[0] + b[2]) / 2)
                center_y = int((b[1] + b[3]) / 2)
                depth = depth_frame.get_distance(center_x, center_y)
                # print(depth)

                # Draw the center point with depth info
                cv2.circle(depth_colormap, (center_x, center_y), radius=5, color=(255, 0, 0), thickness=-1)
                depth_text = f"Depth: {depth:.2f} meters"
                cv2.putText(depth_colormap, depth_text, (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0), 2)

                boundary_depth = []
                for boundary_point in masks[index].xy[0]:  # mask[0] contains the boundary points
                    boundary_x, boundary_y = int(boundary_point[0]), int(boundary_point[1])
                    point_depth = depth_frame.get_distance(boundary_x, boundary_y)
                    boundary_depth.append(point_depth)

                objects_info.append({
                    "class": self.model.names[int(c)],
                    "class_index": c,
                    "box_coordinates": b,
                    "object_center": (center_x, center_y),
                    "center_depth": depth,
                    "boundary_mask": masks[index].xy, #if int(c) == self.target_class_index else None,
                    "boundary_depth": boundary_depth
                })

                annotated_frame = results[0].plot()
                cv2.imshow("color_image", annotated_frame)
                cv2.imshow("depth_image", depth_colormap)
                cv2.imshow("cropped_image", cropped_color_image)

        return objects_info, depth_colormap, cropped_color_image

# only detect object id and pose
def detector():
    model_directory = '/home/yuezang/Desktop/d455/zyyolo8_1.pt'
    Part_detector = ObjectDetector(model_directory)
    all_objects_info = []
    Part_id_list = []
    center_points_list = []
    center_depth_list = []
    masks_list = []
    boundary_depth_list = []

    for _ in range(10):
        all_objects_info.clear()
        Part_id_list.clear()
        center_points_list.clear()
        center_depth_list.clear()
        masks_list.clear()
        boundary_depth_list.clear()
        

        objects_info, color_image, depth_colormap = Part_detector.detect_objects_and_depth()
        # print("#################object_info###############:", objects_info)
        for index in range(len(objects_info)):
            all_objects_info.append(objects_info[index])
            if objects_info[index].get("center_depth", []) > 0:
                print(objects_info[index].get("center_depth", []))
                # make part name as a list
                Part_id = objects_info[index].get("class_index", [])
                Part_id_list.append(Part_id)

                # make center_point as a list
                center_point = objects_info[index].get("object_center", [])
                center_points_list.append(center_point)

                # make filtered depth as a list
                center_depth = objects_info[index].get("center_depth", [])
                center_depth_list.append(center_depth)

                # make a masks list
                masks = objects_info[index].get("boundary_mask", [])
                masks_depth = objects_info[index].get("boundary_depth", [])

                if masks:
                    masks_list.append(masks)
                else:
                    print("No masks found for index", index)

                # make a mask depth list
                if masks_depth:
                    boundary_depth_list.append(copy.deepcopy(masks_depth))
                else:
                    print("No masks_depth found for index", index)

        cv2.imshow("Color Image", color_image)
        cv2.imshow("Depth Image", depth_colormap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    return Part_id_list, center_points_list, center_depth_list, masks_list, boundary_depth_list
    
# calculate the pose from info 
def pose_calculation(center_points_list, center_depth_list, masks_list, boundary_depth_list):
    obj_pose = []
    obj_ori = []
    filtered_masks_lists = None
    for index in range(len(center_points_list)):
            x_value = center_points_list[index][0]
            y_value = center_points_list[index][1]
            depth = center_depth_list[index]
            
            points = masks_list[index][0]
            # print(points)
            if points is None:
                print("No boundary mask available for the last detected object.")
            else:
                width0 = WidthComputer().vectors_method(points, center_points_list[index])
                width1 = WidthComputer().near_far_method(points, center_points_list[index])
                width2 = WidthComputer().box_method(points)
                width = [width0, width1, width2]
                if np.round(np.mean(width)) % 10 >= 5:
                    width_ = np.round(np.mean(width) + (10 - np.round(np.mean(width)) % 10)) * 0.001
                else:
                    width_ = np.round(np.mean(width) - np.round(np.mean(width)) % 10) * 0.001

                point1_uv = np.array([x_value, y_value, 1]).reshape((3, 1))
                center_depth = depth

                calibration_matrix = np.array([[-0.32407458, 0.01029672, 0.9459755, 0.20570073],
                                            [-0.69138456, 0.67994543, -0.24425725, 0.26347288],
                                            [-0.64572676, -0.73319042, -0.21323403, 0.24471352],
                                            [0., 0., 0., 1.]])

                K = np.array([387.31561279296875, 0.0, 317.2518310546875,
                            0.0, 387.31561279296875, 243.16317749023438,
                            0.0, 0.0, 1.0]).reshape((3, 3))

                camera_pose_calculator = CameraPoseCalculator(calibration_matrix, K)
                World_position = camera_pose_calculator.calculate_camera_position(point1_uv, center_depth)
                # print('World position:', World_position)
                obj_pose.append(World_position)

                masks = np.array(points, dtype=np.float32)
                # print(boundary_depth_list[index])
                boundary_depth_ = np.array(boundary_depth_list[index], dtype=np.float32)
                if len(points) == len(boundary_depth_list[index]):
                    boundary_points_3d = np.hstack((masks, boundary_depth_.reshape(-1, 1)))

                    pca_calculator = PCACalculator(n_components=3)
                    transformed_data, principal_components, orientation_angles = pca_calculator.perform_pca_analysis(
                        boundary_points_3d)
                    orientation_angles_ = np.dot(np.linalg.inv(K), orientation_angles)
                    obj_ori.append([orientation_angles_])
                    # all_obj_pose.append(obj_pose)
                    # print(obj_pose)
    return obj_pose, obj_ori


def inspector(state_list):
    model_directory = '/home/yuezang/Desktop/d455/zyyolo8_1.pt'
    Part_detector = ObjectDetector(model_directory)
    all_objects_info = []
    Part_id_list = []
    center_points_list = []
    center_depth_list = []
    masks_list = []
    boundary_depth_list = []

    state_z = []
    # print(state_list)
    for _ in range(len(state_list)):
        print(len(state_list[_]))
        state_z.append(state_list[_][2])
    z_baseline = np.max(state_z)
    print(z_baseline)

    disassmeble = False
    

    # for i in range(10):
    while True:
        # detect_pose.clear()
        for _ in range(10):
            all_objects_info.clear()
            Part_id_list.clear()
            center_points_list.clear()
            center_depth_list.clear()
            masks_list.clear()
            boundary_depth_list.clear()
            

            objects_info, color_image, depth_colormap = Part_detector.detect_objects_and_depth()
            # print("#################object_info###############:", objects_info)
            for index in range(len(objects_info)):
                all_objects_info.append(objects_info[index])
                if objects_info[index].get("center_depth", []) > 0:
                    print(objects_info[index].get("center_depth", []))
                    # make part name as a list
                    Part_id = objects_info[index].get("class_index", [])
                    Part_id_list.append(Part_id)

                    # make center_point as a list
                    center_point = objects_info[index].get("object_center", [])
                    center_points_list.append(center_point)

                    # make filtered depth as a list
                    center_depth = objects_info[index].get("center_depth", [])
                    center_depth_list.append(center_depth)


            cv2.imshow("Color Image", color_image)
            cv2.imshow("Depth Image", depth_colormap)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            detect_pose = []
            for index in range(len(center_points_list)):
                    x_value = center_points_list[index][0]
                    y_value = center_points_list[index][1]
                    depth = center_depth_list[index]
                    point1_uv = np.array([x_value, y_value, 1]).reshape((3, 1))
                    center_depth = depth

                    calibration_matrix = np.array([[-0.32407458, 0.01029672, 0.9459755, 0.20570073],
                                                [-0.69138456, 0.67994543, -0.24425725, 0.26347288],
                                                [-0.64572676, -0.73319042, -0.21323403, 0.24471352],
                                                [0., 0., 0., 1.]])

                    K = np.array([387.31561279296875, 0.0, 317.2518310546875,
                                0.0, 387.31561279296875, 243.16317749023438,
                                0.0, 0.0, 1.0]).reshape((3, 3))

                    camera_pose_calculator = CameraPoseCalculator(calibration_matrix, K)
                    World_position = camera_pose_calculator.calculate_camera_position(point1_uv, center_depth)
                    # print('World position:', World_position)
                    detect_pose.append(World_position)
        print(detect_pose)
        for index in range(len(detect_pose)):
            if detect_pose[index][2] > z_baseline + 0.05:
                print(z_baseline)
                disassmeble = True
                break

        return disassmeble

def disassemble_place_check():
    model_directory = '/home/yuezang/Desktop/d455/zyyolo8_1.pt'
    Part_detector = ObjectDetector(model_directory)
    all_objects_info = []
    Part_id_list = []
    Part_num = 0
    
    # for i in range(10):
    while True:
        # detect_pose.clear()
        for _ in range(10):
            all_objects_info.clear()
            Part_id_list.clear()
            
            objects_info, depth_colormap, cropped_map = Part_detector.detect_in_place()
            # print("#################object_info###############:", objects_info)
            for index in range(len(objects_info)):
                all_objects_info.append(objects_info[index])
                Part_id = objects_info[index].get("class_index", [])
                Part_id_list.append(Part_id)

            cv2.imshow("Color Image", cropped_map)
            cv2.imshow("Depth Image", depth_colormap)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        Part_num += len(Part_id_list)
        if Part_num > 1 or Part_num==0:
            disassemble_too_many = 1
        else:
            disassemble_too_many = 0

        return disassemble_too_many


def init_generation():
    init = []
    init_scenario_condtion = []
    id_list = []
    info4action = []

    Part_name_list, center_points_list, center_depth_list, masks_list, boundary_depth_list = detector()
    # print(len(Part_name_list), len(center_points_list), len(center_depth_list), len(masks_list), len(boundary_depth_list))
    obj_pose, obj_ori = pose_calculation(center_points_list, center_depth_list, masks_list, boundary_depth_list)

    for Part in range(len(Part_name_list)):
        # all_objects_info.append(objects_info[index])
        object_class = Part
        object_class_ = str(object_class)
        if init == []:
            init.append(object_class_)
        elif object_class_ != (init[index] for index in range(len(init))):
            # print(init[index])
            init.append(object_class_)

    init_scenario = list(set(init))
    print(init_scenario)
    # print(init_scenario[0])
    for _ in range(len(init_scenario)):
        scenario_index = float(init_scenario[_])
        # print(scenario_index)
        # Populate the parent dictionary with nested dictionaries
        action_info = {
            "Detected_object_pose": obj_pose[_],
            "Detected_object_ori": obj_ori[_],
            "Destination": [],
            "Actions": []
        }

        id_info = {
            "id": init_scenario[_]
        }

        id_list.append(id_info)
        info4action.append(action_info)
            
        init_scenario_condtion.append({
            str(scenario_index): action_info
        })


    # print("info4action:", info4action)
    # print("init_scenario_condition:", init_scenario_condtion)
        
        
    return id_list, init_scenario_condtion


if __name__ == "__main__":    
    success = disassemble_place_check()
    print(success)


