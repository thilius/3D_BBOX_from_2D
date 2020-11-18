import os
import numpy as np
import cv2
import time
from PIL import Image
import argparse

import torch
from torch.autograd import Variable
import torchvision.transforms as tfs

from model import Net


def get_2d_bbox(filename):
    """
    use yolo to get the 2d bbox
    :param filename: the file for detection
    :return: list, contains the result block of each frame
    """
    pic_ext = ['jpg', 'png', 'jpeg']
    video_ext = ['mp4', 'm4v', 'avi', 'mkv', 'wmv', 'mov']
    
    if filename.split('.')[-1] in video_ext:
        cmds = 'cd C:/Users/wangt/Documents/GITHUB/darknet/build/darknet/x64/ && ' \
               'darknet.exe detector demo 3d/obj.data 3d/vehicle.cfg 3d/vehicle_6000.weights ' \
               '-dont_show -ext_output ' + filename
        res = os.popen(cmds)
        data = res.readlines()
        frames = []
        block = []
        for line in data:
            if 'Objects' in line:
                block = []
            block.append(line)
            if 'FPS' in line:
                frames.append(block)
        filetype = 'video'

    elif filename.split('.')[-1] in pic_ext:
        cmds = 'cd C:/Users/wangt/Documents/GITHUB/darknet/build/darknet/x64/ && ' \
               'darknet.exe detector test 3d/obj.data 3d/vehicle.cfg 3d/vehicle_6000.weights ' \
               '-dont_show -ext_output ' + filename
        res = os.popen(cmds)
        data = res.readlines()
        frames = []
        block = []
        for line in data:
            block.append(line)
        frames.append(block)
        filetype = 'picture'
    else:
        print('Please check again the filename')
        exit(0)

    with open('result.txt', 'w') as f:
        for block in frames:
            f.writelines(block)
    return frames, filetype


def read_labels(index, frames, thresh_2d):
    """
    return the 2d bbox labels of the chosen frame (only the vehicle label)
    :param index: the index of the bbox in the 2d bbox list
    :param frames: list, contains the 2d bbox, each frame has one result block
    :param thresh_2d: only the 2d bbox with the confidence > thresh_2d will be calculated
    :return: the 2d bbox labels [x1, y1, x2, y2, confidence]
    """
    frame = frames[index]
    #the confidence was save as a integer, 30% -> 30, so thresh_2d * 100
    thresh_2d = thresh_2d * 100
    labels = []
    for line in frame:
        label = np.zeros((5,), dtype='int')
        if 'vehicle' in line:
            line = line.split()
            label[0] = int(line[3])
            label[1] = int(line[5])
            label[2] = int(line[7])
            label[3] = int(line[9][:-1])
            label[4] = int(line[1][:-1])
            if label[4] > thresh_2d:
                labels.append(label)
    return labels


def crop_object_img(ori_img, labels):
    """
    crop the object from the original picture
    :param ori_img: the input picture or a frame of the video
    :param labels: 2d bbox matrix, [x1, y1, w, h]
    :return: the cropped picture
    """
    imgs = []
    target_img_size = (224, 224)
    for label in labels:
        #crop the img from the ori_pic
        img = ori_img[label[1]:label[3]+label[1], label[0]:label[2]+label[0]]
        img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(127.5, 127.5, 127.5))
        # add padding to the original picture --> h:w = 1:1
        img = img[..., ::-1]
        h, w, _ = img.shape
        dimension_difference = abs(h - w)
        pad1, pad2 = dimension_difference // 2, dimension_difference - dimension_difference // 2
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        padded_img = np.pad(img, pad, 'constant', constant_values=127.5)
        # save the padded dimensions
        padded_h, padded_w, _ = padded_img.shape
        # resize the picture to the target size1
        resized_img = cv2.resize(padded_img, target_img_size, interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized_img)
        T = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_img = T(pil_img)
        input_img = torch.unsqueeze(input_img, 0)
        imgs.append(input_img)
    return imgs


def load_model(model_path):
    """
    load the 3d generation model from the
    :return: the model with pretrained weights
    """
    model = Net()
    model.load_state_dict(torch.load(model_path))
    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()

    model.eval()
    return model


def get_3d_bbox(imgs, model):
    """
    predict the relative 3d bbox, return the bbox
    :param imgs: the cropped picture
    :param model: the 3d generation model
    :return:
    """
    predictions = []
    labels_3d = []
    for img in imgs:
        input_img = Variable(img.type(torch.cuda.FloatTensor)).cuda()
        with torch.no_grad():
            detections = model(input_img)

        predictions.extend(detections)

    for index, detections in enumerate(predictions):
        xc = detections[..., 0].item()
        yc = detections[..., 1].item()
        w = detections[..., 2].item()
        h = detections[..., 3].item()
        p2x = detections[..., 4].item()
        p2y = detections[..., 5].item()
        p3y = detections[..., 6].item()
        confidence = torch.sigmoid(detections[..., 7]).item()
        label_3d = np.array([xc, yc, w, h, p2x, p2y, p3y, confidence]).reshape(8,)
        labels_3d.append(label_3d)
    return labels_3d


def decode_3d_bbox(labels, labels_3d):
    """
    calculate the real size of the 3d bbox in original picture
    :param labels:  matrix contains the 2d bbox, [x1, y1, w, h]
    :param labels_3d: matrix contains the 3d bbox, [xc, yc, w, h, p2x, p2y, p3x, p3y, confidence], relative size
    :return: labels_3d: matrix contains the 3d bbox, [x1, y1, x2, y2, p2x, p2y, p3x, p3y, confidence], real size in ori_pic
    """
    decoded_bboxes = []
    for i in range(len(labels)):

        ori_xc = labels[i][0] + labels[i][2] / 2.
        ori_yc = labels[i][1] + labels[i][3] / 2.
        if labels[i][2]>labels[i][3]:
            padded_size = labels[i][2]
        else:
            padded_size = labels[i][3]
        #add 40 because we make border in cropped img
        xc = (padded_size + 40) * (labels_3d[i][0] - 0.5) + ori_xc
        yc = (padded_size + 40) * (labels_3d[i][1] - 0.5) + ori_yc
        w = (padded_size + 40) * labels_3d[i][2]
        h = (padded_size + 40) * labels_3d[i][3]
        p2x = (padded_size + 40) * (labels_3d[i][4] - 0.5) + ori_xc
        p2y = (padded_size + 40) * (labels_3d[i][5] - 0.5) + ori_yc
        p3y = (padded_size + 40) * (labels_3d[i][6] - 0.5) + ori_yc
        confidence = labels_3d[i][7]
        x1 = xc - w / 2.
        y1 = yc - h / 2.
        x2 = xc + w / 2.
        y2 = yc + h / 2.

        decoded_bbox = np.array([x1, y1, x2, y2, p2x, p2y, p3y, confidence])
        decoded_bboxes.append(decoded_bbox)
    return decoded_bboxes


def draw_bboxes(draw, labels_2d, bboxes, thresh_3d, size_for_detection):
    """
    draw the 2d and 3d bbox in the original picture, return the drawed picture
    :param draw: the picture for drawing
    :param labels_2d: matrix contains the 2d bbox, [x1, y1, w, h]
    :param bboxes: matrix contains the 3d bbox, [x1, y1, x2, y2, p2x, p2y, p3x, p3y, confidence]
    :param thresh_3d: only the 3d bbox with confidence > thresh_3d will be drawed
    :param size_for_detection: only the 2d bbox with the size > (size_for_detection, size_for_detection) can generate
                                the 3d bbox. smaller size_for_detection, 3d generation for longer distance objects
    """
    thresh_3d = thresh_3d
    size_for_detection = size_for_detection
    for (b, label_2d) in zip(bboxes, labels_2d):
        confidence = b[7]
        if b[4] <= b[0] and confidence > thresh_3d and label_2d[2]>size_for_detection and label_2d[3]>size_for_detection:
            draw = cv2.line(draw, (int(b[4]), int(b[5])), (int(b[0]), int(b[1])), [255, 0, 0], thickness=2)
            draw = cv2.line(draw, (int(b[4]), int(b[6])), (int(b[0]), int(b[3])), [255, 0, 0], thickness=2)
            draw = cv2.line(draw, (int(b[4]), int(b[5])), (int(b[4]), int(b[6])), [255, 0, 0], thickness=2)
            draw = cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), [255, 0, 0], thickness=2)
            draw = cv2.putText(draw, str(int(confidence * 100)), (int(b[0]), int(b[1]) - 2),  cv2.FONT_HERSHEY_PLAIN,
                               1, [255, 0, 0], 1)

        elif b[4] >= b[2] and confidence > thresh_3d and label_2d[2]>size_for_detection and label_2d[3]>size_for_detection:
            draw = cv2.line(draw, (int(b[4]), int(b[5])), (int(b[2]), int(b[1])), [255, 0, 0], thickness=2)
            draw = cv2.line(draw, (int(b[4]), int(b[6])), (int(b[2]), int(b[3])), [255, 0, 0], thickness=2)
            draw = cv2.line(draw, (int(b[4]), int(b[5])), (int(b[4]), int(b[6])), [255, 0, 0], thickness=2)
            draw = cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), [255, 0, 0], thickness=2)
            draw = cv2.putText(draw, str(int(confidence * 100)), (int(b[0]), int(b[1]) - 2), cv2. FONT_HERSHEY_PLAIN,
                               1, [255, 0, 0], 1)

        else:
            draw = cv2.rectangle(draw,
                                 (int(label_2d[0]), int(label_2d[1])),
                                 (int(label_2d[0] + label_2d[2]), int(label_2d[1] + label_2d[3])),
                                 [255, 0, 0], thickness=2)
            draw = cv2.putText(draw, str(int(label_2d[4])), (int(label_2d[0]), int(label_2d[1]) - 2),
                               cv2. FONT_HERSHEY_PLAIN, 1, [255, 0, 0], 1)
    return draw


def get_drawed_picture(index, ori_img, frames, model, thresh_2d, thresh_3d, size_for_detection):
    """
    get the drawed picture with bboxes
    :param index: picture: 0; video: the frame index
    :param ori_img: picture: picture; video: the frame
    :param frames: list, contains the 2d bbox result block
    :param model: the weight-loaded-model for 3d bbox prediction
    :param thresh_2d: 2d bbox with confidence > thresh_2d will be calculated
    :param thresh_3d: 3d bbox with confidence > thresh_3d will be calculated
    :param size_for_detection: the size cropped picture > (size_for_detection, size_for_detection) will generate the 3d bbox
    :return: the drawed picture with 2d/3d bboxes
    """
    # get_2d_bbox
    labels = read_labels(index, frames, thresh_2d)

    if not labels:
        draw = ori_img
    else:
        # make the 3d bbox generation input imgs
        imgs = crop_object_img(ori_img, labels)

        # calculate the 3d bbox
        labels_3d = get_3d_bbox(imgs, model)

        # calculate the 3d bbox in original picture coordinate
        decoded_bboxes = decode_3d_bbox(labels, labels_3d)

        # output the picture with labels
        draw = ori_img.copy()
        draw = draw_bboxes(draw, labels, decoded_bboxes, thresh_3d, size_for_detection)
    return draw


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/epoch_7_train_0.010963_val_0.015008.pth', help='path of the model weight')
    parser.add_argument('--thresh_2d', type=float, default=0.5, help= 'threshold of 2d bbox')
    parser.add_argument('--thresh_3d', type=float, default=0.6, help= 'threshold of 3d bbox')
    parser.add_argument('--size_for_detection', type=float, default=30, help='the min size to generate 3d bbox, affect the distance of 3d bbox detection')
    config = parser.parse_args()
    #parameters
    model_path = config.model_path
    thresh_2d = config.thresh_2d
    thresh_3d = config.thresh_3d
    size_for_detection = config.size_for_detection

    filename = os.path.abspath(input('Please input the filename: ').strip().lower())

    if not os.path.exists(filename):
        print('Please check again the filename')
        exit(0)

    frames, filetype = get_2d_bbox(filename)
    print('get 2d bboxes complete, %s Seconds\n' % (time.time() - start_time))
    tmp_time = time.time()
    print('loading the model........\n')
    model = load_model(model_path)
    print('model loaded, %s Seconds\n\nStarting 3d bbox generation & write the result.......\n' % (time.time() - tmp_time))
    tmp_time = time.time()
    if filetype == 'picture':
        ori_img = cv2.imread(filename)
        draw = get_drawed_picture(0, ori_img, frames, model, thresh_2d, thresh_3d, size_for_detection)
        cv2.imwrite(os.path.splitext(filename)[0] + '_res.jpg', draw)
        print('completed, %s Seconds\n' % (time.time() - tmp_time))
        print('total running time: %s Seconds\n' % (time.time() - start_time))
        os.system(os.path.splitext(filename)[0] + '_res.jpg')


    elif filetype == 'video':
        video = cv2.VideoCapture(filename)
        fps = video.get(cv2.CAP_PROP_FPS)
        size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.splitext(filename)[0] + '_res.avi', fourcc, fps, size)
        i = 0
        while(video.isOpened()):
            ret, frame = video.read()
            if ret == True:
                draw = get_drawed_picture(i, frame, frames, model, thresh_2d, thresh_3d, size_for_detection)
                #cv2.imshow('video', draw)
                #cv2.waitKey(round(1000 / fps))
                out.write(draw)
                i += 1
                if i >= len(frames) - 1:
                    break
            else:
               break

        print('3d bbox generation completed, %s Seconds\n' % (time.time() - tmp_time))
        print('total running time: %s Seconds\n' % (time.time() - start_time))
        video.release()
        out.release()


if __name__ == '__main__':
    main()
