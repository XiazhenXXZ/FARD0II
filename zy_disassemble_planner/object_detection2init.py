import copy
import torch
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

from pca_orientation import PCACalculator
from transformation import CameraPoseCalculator
from width_calculation import WidthComputer

from init_results_detection import detector


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

        return objects_info, color_image, depth_colormap


def init_generation():
    model_directory = '/home/yuezang/Desktop/d455/zyyolo8_1.pt'
    # model_directory = '/home/yuezang/Desktop/d455/yolov8_medbin_v0/Medbin_YOLO_v3.pt'
    # target_class_index = 27
    MW_detector = ObjectDetector(model_directory)#, target_class_index)
    init = []
    init_scenario_condtion = []
    id_list = []
    info4action = []

    for _ in range(10):
        objects_info, color_image, depth_colormap = MW_detector.detect_objects_and_depth()
        # print(objects_info)
        for index in range(len(objects_info)):
            # all_objects_info.append(objects_info[index])
            object_class = objects_info[index].get("class_index", [])
            object_class_ = str(object_class)
            if init == []:
                init.append(object_class_)
            elif object_class_ != (init[index] for index in range(len(init))):
                # print(init[index])
                init.append(object_class_)
            else:
                # break
                MW_detector.pipeline.stop()
        

        cv2.imshow("Color Image", color_image)
        cv2.imshow("Depth Image", depth_colormap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    MW_detector.pipeline.stop()
    cv2.destroyAllWindows()

    init_scenario = list(set(init))
    print(init_scenario)

    for _ in range(len(init_scenario)):
        scenario_index = float(init_scenario[_].split('[')[1].split(']')[0])
        # print(scenario_index)
        MW_pose = detector(scenario_index)

        # Populate the parent dictionary with nested dictionaries
        action_info = {
            "Detected_object_information": MW_pose,
            "Destination": [],
            "Actions": []
        }

        id_info = {
            "id": scenario_index
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
    # id_init, condition_init = init_generation()
    
    # print("id_init:", id_init)
    # print("condition_init:", condition_init)
    id, init = init_generation()
    print("id:", id)
    print("init", init)
