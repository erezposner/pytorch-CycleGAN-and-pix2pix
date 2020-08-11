import torch.nn as nn
import torch
import cv2
import numpy as np
import dlib
from imutils import face_utils
import os
from pathlib import Path
from Gavros.Utils.Utils import download_if_not_exists

class LandmarksDlibFE():

    def __init__(self, weights_path='FeatureExtraction/LandMarkDlib/resources/shape_predictor_68_face_landmarks.dat'):
        super(LandmarksDlibFE, self).__init__()

        shape_predictor_68_face_landmarks_utl ='https://drive.google.com/uc?export=download&id=1nhOe9ktLcMwY28VpOw4ki1JWI8hRW_ct'
        download_if_not_exists(weights_path,shape_predictor_68_face_landmarks_utl,approx_total_size_kb=1e+8)

        self.face_detector = dlib.get_frontal_face_detector()

        self.face_landmarks_predictor = dlib.shape_predictor(weights_path)


    def dlib_get_landmarks(self, target_img, rect, face_landmarks_predictor):
        '''
        If rect is none also calls the predictor, otherwise only calls the landmarks detector
            (significantly faster)
        '''
        gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        shape = face_landmarks_predictor(gray, rect)
        # landmarks2D = face_utils.shape_to_np(shape)[17:]
        landmarks2D = face_utils.shape_to_np(shape)
        # Mirror landmark y-coordinates
        landmarks2D[:, 1] = target_img.shape[0] - landmarks2D[:, 1]
        return landmarks2D

    def dlib_get_face_rectangle(self, target_img, face_detector):
        '''
        If rect is none also calls the predictor, otherwise only calls the landmarks detector
            (significantly faster)
        '''
        gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        rects = face_detector(gray, 0)
        # if (len(rects) == 0):
        #     print('Error: could not locate face')
        return rects[0]

    def __call__(self, **kwargs):
        def forward(*args):
            return self.get_2d_lmks(*args)

        return forward(kwargs['image'])


    def get_2d_lmks(self, image: np.ndarray) -> np.ndarray:

        rect = self.dlib_get_face_rectangle(image, self.face_detector)
        target_2d_lmks = self.dlib_get_landmarks(image, rect, self.face_landmarks_predictor)
        target_2d_lmks[:, 1] = image.shape[1] - target_2d_lmks[:, 1]  # revert Y axis
        return target_2d_lmks

    def display(self, image, landmarks2D, output_path, base_name):

        for (x, y) in landmarks2D:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)

        return self.save(output_path, base_name, image)

    def save(self, output_folder, output_basename, input_image):
        filename, file_extension = os.path.splitext(output_basename)



        image_oj_path = str(Path(output_folder) / (filename + '_landmarks_dlib.png'))
        cv2.imwrite(str(image_oj_path), input_image[..., ::-1])
        return image_oj_path
