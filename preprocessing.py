# -*- coding: utf-8 -*-
'''
According to the paper, the authors extracted upto 80 frames from each video,
they did not mention if they grabbed first 80 frames, or sampled 80 frames with same intervals,
but anyway I did the latter.
'''
import cv2
import os
# import ipdb
import numpy as np
import pandas as pd
import skimage
from cnn_util import *
from parameters import *


def preprocess_frame(image, target_height=227, target_width=227):
    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    image = skimage.img_as_float(image).astype(np.float32)
    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height, target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height) / height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:, cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length, :]

    return cv2.resize(resized_image, (target_height, target_width))


def main():
    # num_frames = 80
    # caffe_root = '/Users/Udit/programs/github/caffe'
    # vgg_model = caffe_root + '/models/vgg/VGG_ILSVRC_19_layers.caffemodel'
    # vgg_deploy = caffe_root + '/models/vgg/VGG_ILSVRC_19_layers_deploy.prototxt'
    # # video_path = '/Users/Udit/Downloads/Datasets for ML FP/YouTubeClips'
    # video_save_path = '/Users/Udit/Downloads/sample_video/save'
    # video_path = '/Users/Udit/Downloads/sample_video'
    videos = os.listdir(video_path)
    videos = filter(lambda x: x.endswith('mp4'), videos)

    cnn = CNN()
    count = 1
    for video in videos:
        print video
        count += 1
        if (count > 5):
            break
        # print video

        if os.path.exists( os.path.join(video_features_path, video) ):
            print "Already processed ... "
            continue

        video_fullpath = os.path.join(video_path, video)
        try:
            print video_fullpath
            cap  = cv2.VideoCapture( video_fullpath )
        except:
            print 'cap not available'
            pass

        frame_count = 0
        frame_list = []

        while True:
            ret, frame = cap.read()
            if ret is False:
                print 'Failed to read video (mp4)'
                break

            frame_list.append(frame)
            frame_count += 1

        frame_list = np.array(frame_list)

        if frame_count > 80:
            frame_indices = np.linspace(0, frame_count, num=num_frames, endpoint=False).astype(int)
            frame_list = frame_list[frame_indices]

        cropped_frame_list = np.array(map(lambda x: preprocess_frame(x), frame_list))
        feats = cnn.get_features(cropped_frame_list)
        # print feats
        save_full_path = os.path.join(video_features_path, video + '.npy')
        np.save(save_full_path, feats)


if __name__ == "__main__":
    main()
