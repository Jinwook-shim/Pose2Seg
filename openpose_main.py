import os
import json
import glob
import numpy as np
# import cv2
import matplotlib.pyplot as plt
from pathlib import Path

class OpenPoseRunner():
    def __init__(self, release_path, data_folder, output_folder):
        self.release_path = release_path
        self.data_folder = data_folder

        self.output_folder = output_folder
        Path(self.output_folder).mkdir(parents=True,exist_ok=True)
        #         self.command = release_path + r'\bin\OpenPoseDemo.exe --image_dir ' + self.data_folder + ' --hand --hand_detector 3  --write_json ' + self.output_folder + ' --write_images ' + self.output_folder
        self.command = release_path + r'\bin\OpenPoseDemo.exe --image_dir ' + self.data_folder + '  --hand --hand_scale_number 6 --hand_scale_range 0.4 --hand_detector 3  --write_json ' + self.output_folder + ' --display 0 --write_images ' + self.output_folder
        self.command = release_path + r'\bin\OpenPoseDemo.exe --image_dir ' + self.data_folder + '  --hand --hand_scale_number 6 --hand_scale_range 0.4 --hand_detector 3  --write_json ' + self.output_folder + ' --model_pose COCO --display 0 --write_images ' + self.output_folder

    #         self.command = release_path + r'\bin\OpenPoseDemo.exe --image_dir ' + self.data_folder + '  --hand --write_json ' + self.output_folder

    def runOpenPose(self):
        current_folder = os.getcwd()
        os.chdir(self.release_path)
        os.system(self.command)
        os.chdir(current_folder)

    #         self.ReArragne_keypoints_indices(self.output_folder)

    def ReArragne_keypoints_indices(self, output_folder):
        coco_indices = np.array([range(0, 21)])
        rearrange_finger_indices = np.array([0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17])

        for idx, file in enumerate(glob.glob(self.output_folder + "\*.json")):
            file_name = file.split('\\')[-1]
            #             print('{} - {}'.format(idx, file_name))

            #         print(file_name)
            #         print(file)
            ms = 10
            with open(file, 'r') as f:
                datastore = json.load(f)
                num_people = len(datastore['people'])
                for personID in range(0, num_people):
                    left_hand = datastore['people'][personID]['hand_left_keypoints_2d']

                    left_hand = np.reshape(np.array(left_hand), (-1, 3))
                    left_hand = left_hand[rearrange_finger_indices, :]
                    right_hand = datastore['people'][personID]['hand_right_keypoints_2d']
                    right_hand = np.reshape(np.array(right_hand), (-1, 3))
                    right_hand = right_hand[rearrange_finger_indices, :]
                    right_hand = right_hand.flatten().tolist()
                    left_hand = left_hand.flatten().tolist()

                    datastore['people'][personID]['hand_left_keypoints_2d'] = left_hand
                    datastore['people'][personID]['hand_right_keypoints_2d'] = right_hand

            #             with open(self.output_folder + '\modified_' + file_name, 'w') as f:
            with open(self.output_folder + '\\' + file_name, 'w') as f:
                json.dump(datastore, f)
        print('done rearragning hand keypoints indices')
        print('')


if __name__ == "__main__":
    release_path = r'C:\Erez.Posner_to_NAS\openpose\release\openpose-1.5.0-binaries-win64-gpu-python-flir-3d_recommended'
    data_folder = release_path + r'\examples\media'
    data_folder = r'\\fs01\Algo\ML\Datasets\NeuralPCL\Data\Adaya_2019-11-05-16.41.21_joined\Components\RGB'
    output_folder = r'\\fs01\Algo\ML\Datasets\NeuralPCL\Data\Adaya_2019-11-05-16.41.21_joined\Components\coco_skeleton'
    op = OpenPoseRunner(release_path, data_folder, output_folder)
    op.runOpenPose()
    # op.ReArragne_keypoints_indices(output_folder)
