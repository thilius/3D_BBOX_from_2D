import cv2
import numpy as np
import random


def show_labeled_pic(file_path, target_size=600):
    ''' visulize the made pictures with bbox
        target size: the target size for showing
        file_path: the path contains the train/val/test filename
    '''
    #get a random picture from the filelist
    with open(file_path) as f:
        data = f.readlines()
    index = random.randint(0, len(data))
    pic_path = data[index].rstrip()
    print(pic_path)
    img = cv2.imread(pic_path)
    #convert bgr to rgb
    #img = img[..., ::-1]
    img = np.array(img)

    #add padding to the original picture --> h:w = 1:1
    h, w, _ = img.shape
    dimension_difference = np.abs(h - w)
    pad1, pad2 = dimension_difference//2, dimension_difference - dimension_difference//2
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h<=w else ((0, 0), (pad1, pad2), (0, 0))
    padded_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255


    #save the padded dimensions
    padded_h, padded_w, _ = padded_img.shape
    #resize the picture to the target size
    resized_img = cv2.resize(padded_img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    labels = np.loadtxt(pic_path.replace('jpg', 'txt').replace('png', 'txt')).reshape(-1, 8)

    #calculate the dimensions of the original picture
    x1 = (labels[:, 0] - labels[:, 2]/2) * w
    y1 = (labels[:, 1] - labels[:, 3]/2) * h
    x2 = (labels[:, 0] + labels[:, 2]/2) * w
    y2 = (labels[:, 1] + labels[:, 3]/2) * h
    px = (labels[:, 4]) * w
    py = (labels[:, 5]) * h
    p3y = (labels[:, 6]) * h
    #add the padding to the original dimensions
    x1 += pad[1][0]
    y1 += pad[0][0]
    x2 += pad[1][0]
    y2 += pad[0][0]
    px += pad[1][0]
    py += pad[0][0]
    p3y += pad[0][0]
    #recalculate the dimensions on the new resized picture
    labels[:, 0] = ((x1+x2)/2) / padded_w
    labels[:, 1] = ((y1+y2)/2) / padded_h
    labels[:, 2] *= w / padded_w
    labels[:, 3] *= h / padded_h
    labels[:, 4] = px / padded_w
    labels[:, 5] = py / padded_h
    labels[:, 6] = p3y / padded_h

    #show the resized labels on the resized picture
    x1 = (labels[:, 0] - labels[:, 2]/2) * target_size
    y1 = (labels[:, 1] - labels[:, 3]/2) * target_size
    x2 = (labels[:, 0] + labels[:, 2]/2) * target_size
    y2 = (labels[:, 1] + labels[:, 3]/2) * target_size
    p2x = labels[:, 4] * target_size
    p2y = labels[:, 5] * target_size
    p3x = p2x
    p3y = labels[:, 6] * target_size
    labeled_img = cv2.rectangle(resized_img, (x1, y1), (x2, y2), [255, 0, 0])
    labeled_img = cv2.line(labeled_img, (p2x, p2y), (p3x, p3y), [255, 0, 0])
    if p2x < x1:
        labeled_img = cv2.line(labeled_img, (p2x, p2y), (x1, y1), [255, 0, 0])
        labeled_img = cv2.line(labeled_img, (p3x, p3y), (x1, y2), [255, 0, 0])
    else:
        labeled_img = cv2.line(labeled_img, (p2x, p2y), (x2, y1), [255, 0, 0])
        labeled_img = cv2.line(labeled_img, (p3x, p3y), (x2, y2), [255, 0, 0])
    cv2.imshow(pic_path, labeled_img)
    cv2.waitKey(2000)
    cv2.destroyWindow(pic_path)

if __name__ == '__main__':
    while True:
        show_labeled_pic('C:/Users/wangt/PycharmProjects/3d/train.txt')

