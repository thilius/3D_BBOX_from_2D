import os
import numpy as np
import cv2
import time
from PIL import Image
import argparse
import math
import json

import torch
from torch.autograd import Variable
import torchvision.transforms as tfs

from model import Net


JSONFILE = {'2D': {}, '3D': {}}

def get_2d_bbox(filename):
    """
    use yolo to get the 2d bbox, only mp4, avi, jpg, png file will be detected
    :param filename: the file path for detection
    :return: list, contains the result block of each frame, also save the YOLO result as result.txt
    """
    
    pic_ext = ['jpg', 'png', 'jpeg']
    video_ext = ['mp4', 'm4v', 'avi', 'mkv', 'wmv', 'mov']
    
    if filename.split('.')[-1] in video_ext:
        cmds = 'cd C:/Users/wangt/Documents/darknet/build/darknet/x64/ && ' \
               'darknet.exe detector demo bdd/obj.data bdd/bdd.cfg bdd/bdd.weights ' \
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
        cmds = 'cd C:/Users/wangt/Documents/darknet/build/darknet/x64/ && ' \
               'darknet.exe detector test bdd/obj.data bdd/bdd.cfg bdd/bdd.weights ' \
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

    with open('%s_result.txt' % os.path.splitext(os.path.basename(filename))[0], 'w') as f:
        for block in frames:
            f.writelines(block)
    return frames, filetype


def read_labels(index, frames, thresh_2d):
    """
    get the 2d bbox labels from the YOLO result respectively : [x1, y1, width, height, confidence]
    :param index: int, the index of the bbox in the bbox list
    :param frames: list of blocks, contains the 2d bbox, each frame has one result block
    :param thresh_2d: bbox confidence < thresh_2d will be eliminated
    :return: vehicle bbox = labels_for_3d, for further 3d bbox generation, other bbox = labels_2d
    """
    frame = frames[index]
    JSONFILE['2D'][index] = {}
    JSONFILE['2D'][index]['vehicle'] = []
    JSONFILE['2D'][index]['cycle'] = []
    JSONFILE['2D'][index]['human'] = []
    # the confidence was save as a integer, 30% -> 30, so thresh_2d * 100
    thresh_2d *= 100
    labels_for_3d = []
    labels_2d = []
    for line in frame:
        label = np.zeros((5,), dtype='int')
        if 'vehicle' in line:
        # sometimes there's some wrong detection with two categories in one line
            if ',' in line:
                line = line.split(',')[1]
            line = line.split()
            label[0] = int(line[3])
            label[1] = int(line[5])
            label[2] = int(line[7])
            label[3] = int(line[9][:-1])
            label[4] = int(line[1][:-1])
            JSONFILE['2D'][index]['vehicle'].append(label.tolist())
            if label[4] > thresh_2d:
                labels_for_3d.append(label)
        elif 'cycle' in line:
            # sometimes there's some wrong detection with two categories in one line
            if ',' in line:
                line = line.split(',')[1]
            line = line.split()
            label[0] = int(line[3])
            label[1] = int(line[5])
            label[2] = int(line[7])
            label[3] = int(line[9][:-1])
            label[4] = int(line[1][:-1])
            JSONFILE['2D'][index]['cycle'].append(label.tolist())
            if label[4] > thresh_2d:
                labels_2d.append(label)
        elif 'human' in line:
            # sometimes there's some wrong detection with two categories in one line
            if ',' in line:
                line = line.split(',')[1]
            line = line.split()
            label[0] = int(line[3])
            label[1] = int(line[5])
            label[2] = int(line[7])
            label[3] = int(line[9][:-1])
            label[4] = int(line[1][:-1])
            JSONFILE['2D'][index]['human'].append(label.tolist())
            if label[4] > thresh_2d:
                labels_2d.append(label)
   
    return labels_for_3d, labels_2d


def crop_object_img(ori_img, labels):
    """
    crop the objects from the original picture, also do the pretreatment for the picture
    :param ori_img: the input picture or a frame of the video
    :param labels: 2d bbox matrix for 3d bbox generation, [x1, y1, width, height, confidence]
    :return: the cropped picture as pytorch Tensor
    """
    imgs = []
    target_img_size = (224, 224)
    for label in labels:
        # crop the img from the ori_pic
        img = ori_img[label[1]:label[3] + label[1], label[0]:label[2] + label[0]]
        img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(127.5, 127.5, 127.5))
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
    load the model for 3d bbox generation
    :param model_path: path of the weight file
    :return: the model with pretrained weight
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
    get the prediction
    :param imgs: the cropped picture
    :param model: the 3d generation model
    :return: list, contains the encoded 3d bbox as numpy array
    """
    predictions = []

    for img in imgs:
        input_img = Variable(img.type(torch.cuda.FloatTensor)).cuda()
        with torch.no_grad():
            prediction = model(input_img)
            prediction = prediction.cpu().numpy()

        predictions.extend(prediction)
    return predictions


def decode_3d_bbox(labels_for_3d, labels_3d):
    """
    transform the prediction into the original picture coordinate
    :param labels_for_3d: list, contains the 2d bbox, [x1, y1, width, height, confidence]
    :param labels_3d: list, contains the prediction
    :return: transformed prediction
    """
    decoded_bboxes = []
    for i in range(len(labels_for_3d)):
        ori_xc = labels_for_3d[i][0] + labels_for_3d[i][2] / 2.
        ori_yc = labels_for_3d[i][1] + labels_for_3d[i][3] / 2.
        padded_size = max(labels_for_3d[i][2], labels_for_3d[i][3])

        # add 20 because we make border in cropped img

        decoded_bbox = labels_3d[i].reshape(10, )

        decoded_bbox[0] = (decoded_bbox[0] - 0.5) * (padded_size + 20) + ori_xc
        decoded_bbox[1] = (decoded_bbox[1] - 0.5) * (padded_size + 20) + ori_yc
        decoded_bbox[3] *= (padded_size + 20)
        decoded_bbox[4] *= (padded_size + 20)
        decoded_bbox[5] = (decoded_bbox[5] - 0.5) * (padded_size + 20) + ori_xc
        decoded_bbox[6] = (decoded_bbox[6] - 0.5) * (padded_size + 20) + ori_yc
        decoded_bbox[8] *= (padded_size + 20)
        decoded_bbox[9] *= (padded_size + 20)

        decoded_bboxes.append(decoded_bbox)

    return decoded_bboxes


def decode_coordinate(decoded_bboxes, index):
    """
    get the coordinate of the parallelogram of the front and back face
    :param decoded_bboxes: the transformed prediction in original picture coordinate
    :return: the coordinate of the parallelogram of the front and back face
    """
    decoded_coors = []
    JSONFILE['3D'][index] = []
    for decoded_bbox in decoded_bboxes:
        mid_f_1_x = decoded_bbox[0] - decoded_bbox[3] / 2.
        mid_f_2_x = decoded_bbox[0] + decoded_bbox[3] / 2.
        mid_f_1_y = decoded_bbox[1] - math.tan(decoded_bbox[2]) * decoded_bbox[3] / 2.
        mid_f_2_y = decoded_bbox[1] + math.tan(decoded_bbox[2]) * decoded_bbox[3] / 2.
        x1 = mid_f_1_x
        x5 = mid_f_1_x
        x2 = mid_f_2_x
        x6 = mid_f_2_x
        y1 = mid_f_1_y + decoded_bbox[4] / 2.
        y5 = mid_f_1_y - decoded_bbox[4] / 2.
        y2 = mid_f_2_y + decoded_bbox[4] / 2.
        y6 = mid_f_2_y - decoded_bbox[4] / 2.

        mid_b_1_x = decoded_bbox[5] - decoded_bbox[8] / 2.
        mid_b_2_x = decoded_bbox[5] + decoded_bbox[8] / 2.
        mid_b_1_y = decoded_bbox[6] - math.tan(decoded_bbox[7]) * decoded_bbox[8] / 2.
        mid_b_2_y = decoded_bbox[6] + math.tan(decoded_bbox[7]) * decoded_bbox[8] / 2.
        x4 = mid_b_1_x
        x8 = mid_b_1_x
        x3 = mid_b_2_x
        x7 = mid_b_2_x
        y4 = mid_b_1_y + decoded_bbox[9] / 2.
        y8 = mid_b_1_y - decoded_bbox[9] / 2.
        y3 = mid_b_2_y + decoded_bbox[9] / 2.
        y7 = mid_b_2_y - decoded_bbox[9] / 2.
        decoded_coor = np.array([[x1, x2, x6, x5, x4, x3, x7, x8], [y1, y2, y6, y5, y4, y3, y7, y8]]).astype(
            int).reshape(
            (2, 8))
        decoded_coors.append(decoded_coor)

        JSONFILE['3D'][index].append(decoded_coor.tolist())
    return decoded_coors


def draw_decoded_bbox(draw, decoded_coor):
    """
    draw the 3d bbox
    :param draw: the picture for drawing
    :param decoded_coor: the coordinate of the parallelogram of the front and back face
    :return: the drawed picture
    """
    idx = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 4, 7, 3], [1, 5, 6, 2]]).reshape((4, 4))
    if len(decoded_coor) != 0:
        for f in range(3, -1, -1):
            pts = []
            for j in range(4):
                point = [int(decoded_coor[0, idx[f, j]]) + 1, int(decoded_coor[1, idx[f, j]]) + 1]
                pts.append(point)
            pts = np.array(pts).reshape((-1, 1, 2))
            if f == 0:
                cv2.polylines(draw, [pts], isClosed=True, thickness=2, color=[0, 0, 255])
            elif f == 3:
                cv2.polylines(draw, [pts], isClosed=True, thickness=2, color=[0, 255, 0])
            else:
                cv2.polylines(draw, [pts], isClosed=True, thickness=2, color=[0, 255, 0])
    return draw


def draw_bboxes(draw, labels_2d, labels_for_3d, decoded_coors, thresh_3d, size_for_detection):
    """
    draw the bboxes in the original picture
    :param draw: the picture for drawing
    :param labels_2d: the 2d bbox of other categories
    :param labels_for_3d: the 2d bbox of vehicles
    :param decoded_coors: the coordinate of the parallelogram of the front and back face
    :param thresh_3d: here not used
    :param size_for_detection: determine the minimal size for 3d bbox detection, > (size_for_detection, size_for_detection)
    :return: the picture with the bboxes
    """
    thresh_3d = thresh_3d
    size_for_detection = size_for_detection
    for (decoded_coor, label_for_3d) in zip(decoded_coors, labels_for_3d):
        if label_for_3d[2] > size_for_detection and label_for_3d[3] > size_for_detection:
            draw = draw_decoded_bbox(draw, decoded_coor)
        else:
            draw = cv2.rectangle(draw,
                                 (int(label_for_3d[0]), int(label_for_3d[1])),
                                 (int(label_for_3d[0] + label_for_3d[2]), int(label_for_3d[1] + label_for_3d[3])),
                                 [255, 0, 0], thickness=2)
            draw = cv2.putText(draw, str(int(label_for_3d[4])), (int(label_for_3d[0]), int(label_for_3d[1]) - 2),
                               cv2.FONT_HERSHEY_PLAIN, 1, [255, 0, 0], 1)
    for label_2d in labels_2d:
        draw = cv2.rectangle(draw,
                             (int(label_2d[0]), int(label_2d[1])),
                             (int(label_2d[0] + label_2d[2]), int(label_2d[1] + label_2d[3])),
                             [255, 0, 0], thickness=2)
        draw = cv2.putText(draw, str(int(label_2d[4])), (int(label_2d[0]), int(label_2d[1]) - 2),
                           cv2.FONT_HERSHEY_PLAIN, 1, [255, 0, 0], 1)

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
    labels_for_3d, labels_2d = read_labels(index, frames, thresh_2d)

    if not labels_for_3d and not labels_2d:
        draw = ori_img
    else:
        # make the 3d bbox generation input imgs
        imgs = crop_object_img(ori_img, labels_for_3d)

        # calculate the 3d bbox
        labels_3d = get_3d_bbox(imgs, model)

        # calculate the 3d bbox in original picture coordinate
        decoded_bboxes = decode_3d_bbox(labels_for_3d, labels_3d)

        decoded_coors = decode_coordinate(decoded_bboxes, index)

        # output the picture with labels
        draw = ori_img.copy()
        draw = draw_bboxes(draw, labels_2d, labels_for_3d, decoded_coors, thresh_3d, size_for_detection)
    return draw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/kitti.pth',
                        help='path of the model weight')
    parser.add_argument('--thresh_2d', type=float, default=0.7, help='threshold of 2d bbox')
    parser.add_argument('--thresh_3d', type=float, default=0.5, help='threshold of 3d bbox')
    parser.add_argument('--size_for_detection', type=float, default=30,
                        help='the min size to generate 3d bbox, affect the distance of 3d bbox detection')
    config = parser.parse_args()
    # parameters
    model_path = config.model_path
    thresh_2d = config.thresh_2d
    thresh_3d = config.thresh_3d
    size_for_detection = config.size_for_detection

    filename = os.path.abspath(input('Please input the filename: ').strip().lower())

    start_time = time.time()

    if not os.path.exists(filename):
        print('Please check again the filename')
        exit(0)

    frames, filetype = get_2d_bbox(filename)
    print('get 2d bboxes complete, %s Seconds\n' % (time.time() - start_time))

    tmp_time = time.time()
    print('loading the model........\n')
    model = load_model(model_path)
    print('model loaded, %s Seconds\n\nStarting 3d bbox generation & write the result.......\n' % (
            time.time() - tmp_time))
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
        while (video.isOpened()):
            ret, frame = video.read()
            if ret == True:
                draw = get_drawed_picture(i, frame, frames, model, thresh_2d, thresh_3d, size_for_detection)
                # cv2.imshow('video', draw)
                # cv2.waitKey(round(1000 / fps))
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
    
    with open(os.path.splitext(filename)[0] + '.json', 'w', encoding='utf-8') as f:
        json.dump(JSONFILE, f)


if __name__ == '__main__':
    main()
