import numpy as np
import os
import cv2
import random
from tqdm import tqdm
import math

root_dir = 'D:/dataset/KITTI/training/'


def get_filepath(root_dir):
    calib_filepathes = []
    filepathes = os.listdir(root_dir + 'calib')
    for file in filepathes:
        file = root_dir + 'calib/' + file
        calib_filepathes.append(file)
    random.shuffle(calib_filepathes)
    return calib_filepathes


def readCalibration(calib_filepath):
    with open(calib_filepath) as f:
        data = f.readlines()
    p2 = data[2].split()[1:]
    p2 = np.array(p2).astype(float).reshape([3, 4])
    return p2


def readLabels(calib_filepath):
    objs = []
    filepath = calib_filepath.replace('calib', 'label_2')
    with open(filepath) as f:
        data = f.readlines()
    for line in data:
        if line[0] == 'D':
            continue
        obj = line.split()
        objs.append(obj)
    return objs


def compute3dBBox(obj, p2):
    face_idx = np.array([[0, 1, 5, 4],  # front face
                         [1, 2, 6, 5],  # right face
                         [2, 3, 7, 6],  # back face
                         [3, 0, 4, 7]])  # left face
    ry = float(obj[14])
    rotation_matrix = np.array([np.cos(ry), 0, np.sin(ry), 0, 1, 0, -np.sin(ry), 0, np.cos(ry)]).reshape((3, 3))
    height, width, length = float(obj[8]), float(obj[9]), float(obj[10])
    trans_x, trans_y, trans_z = float(obj[11]), float(obj[12]), float(obj[13])

    x_corners = np.array(
        [length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2, -length / 2, -length / 2])
    y_corners = np.array([0, 0, 0, 0, -height, -height, -height, -height])
    z_corners = np.array([width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2])

    corners_3d = np.stack([x_corners, y_corners, z_corners], axis=0)
    corners_3d = np.matmul(rotation_matrix, corners_3d)
    corners_3d[0, :] += trans_x
    corners_3d[1, :] += trans_y
    corners_3d[2, :] += trans_z
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = []
    else:
        corners_2d = projectToImage(corners_3d, p2)
    return corners_2d, face_idx


def projectToImage(pts_3d, p2):
    addon = np.ones([1, pts_3d.shape[1]])
    pts_3d = np.concatenate([pts_3d, addon], axis=0)
    # project in image
    pts_2d = np.matmul(p2, pts_3d)
    # scale projected points
    pts_2d[0, :] /= pts_2d[2, :]
    pts_2d[1, :] /= pts_2d[2, :]
    pts_2d = np.delete(pts_2d, 2, axis=0)
    return pts_2d


def computeOrientation3d(obj, p2):
    # compute rotational matrix around yaw axis
    ry = float(obj[14])
    rotation_matrix = np.array([np.cos(ry), 0, np.sin(ry), 0, 1, 0, -np.sin(ry), 0, np.cos(ry)]).reshape((3, 3))
    # compute in obj coordinate system
    orientation_3d = np.array([0., float(obj[10]), 0., 0., 0., 0.]).reshape((3, 2))

    # rotate and translate in camera coordinate system, project in image
    trans_x, trans_y, trans_z = float(obj[11]), float(obj[12]), float(obj[13])
    orientation_3d = np.matmul(rotation_matrix, orientation_3d)
    orientation_3d[0, :] += trans_x
    orientation_3d[1, :] += trans_y
    orientation_3d[2, :] += trans_z

    # vector behind image plane?
    if np.any(orientation_3d[2, :] < 0.1):
        orientation_2d = []
    else:
        orientation_2d = projectToImage(orientation_3d, p2)
    return orientation_2d


def draw3dBBox(img, obj, corners, face_idx, orientation):
    occ_col = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    print(corners.astype(int))
    if corners != []:
        for f in range(0, 4):
            pts = []
            for j in range(4):
                point = [int(corners[0, face_idx[f, j]]) + 1, int(corners[1, face_idx[f, j]]) + 1]
                pts.append(point)
            pts = np.array(pts).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, thickness=2, color=occ_col[int(obj[2])])
    return img


def show_bbox():
    calib_filepaths = get_filepath(root_dir)
    i = 0
    while 0 <= i < len(calib_filepaths):
        calib_filepath = calib_filepaths[i]
        p2 = readCalibration(calib_filepath)
        objs = readLabels(calib_filepath)
        pic_path = calib_filepath.replace('calib', 'image_2').replace('txt', 'png')
        img = cv2.imread(pic_path)
        draw = img.copy()
        for obj in objs:
            corners_2d, face_idx = compute3dBBox(obj, p2)
            orientation_2d = computeOrientation3d(obj, p2)
            draw = draw3dBBox(draw, obj, corners_2d, face_idx, orientation_2d)
        cv2.putText(draw, 'ESC: Quit, U: prev picture, Any other: next picture', (2, 10), cv2.FONT_HERSHEY_PLAIN,
                    1, [0, 255, 0], 2)
        cv2.imshow(pic_path, draw)
        key = cv2.waitKey(0)
        if key == 27:
            break
        elif key == 117:
            i -= 1
            cv2.destroyAllWindows()
            continue
        else:
            cv2.destroyAllWindows()
            i += 1
            continue


def make_trainfile_v1():
    vehicle_cata = ['Car', 'Van', 'Tram', 'Misc', 'Truck']
    calib_filepaths = get_filepath(root_dir)
    os.makedirs('data', exist_ok=True)

    i = 0
    calib_filepaths = tqdm(calib_filepaths)
    for calib_filepath in calib_filepaths:
        p2 = readCalibration(calib_filepath)
        objs = readLabels(calib_filepath)
        pic_path = calib_filepath.replace('calib', 'image_2').replace('txt', 'png')
        img = cv2.imread(pic_path)

        for obj in objs:
            if obj[0] not in vehicle_cata:
                continue

            x1_2d = float(obj[4])
            y1_2d = float(obj[5])
            x2_2d = float(obj[6])
            y2_2d = float(obj[7])

            if (x2_2d - x1_2d) < 30 or (y2_2d - y1_2d) < 30:
                continue

            cropped_img = img[int(y1_2d): int(y2_2d), int(x1_2d): int(x2_2d)]
            cropped_img = cv2.copyMakeBorder(cropped_img, 20, 20, 20, 20, cv2.BORDER_CONSTANT,
                                             value=(127.5, 127.5, 127.5))

            corners, face_idx = compute3dBBox(obj, p2)
            if corners == []:
                continue

            corners[0, :] = (corners[0, :] - x1_2d + 20) / cropped_img.shape[1]
            corners[1, :] = (corners[1, :] - y1_2d + 20) / cropped_img.shape[0]
            corners = corners.reshape(16, )

            truncated = float(obj[1])
            occluded = int(obj[2])

            if truncated > 0.2 or occluded > 1:
                bias = 0.
            else:
                bias = 1.
            tcorners = np.append(corners, bias)
            if i < 23400:
                cv2.imwrite('data/train_%d.jpg' % i, cropped_img)
                np.savetxt('data/train_%d.txt' % i, tcorners, fmt='%.6f')
            else:
                cv2.imwrite('data/valid_%d.jpg' % i, cropped_img)
                np.savetxt('data/valid_%d.txt' % i, tcorners, fmt='%.6f')
            i += 1

    with open('train.txt', 'w') as f:
        for j in range(23400):
            f.write('data/train_%d.jpg\n' % j)
    with open('valid.txt', 'w') as f:
        for j in range(23400, i):
            f.write('data/valid_%d.jpg\n' % j)


def encode_coordinate(corners, rotation):
    if rotation < 0:
        mid_f_1_x = (corners[0, 0] + corners[0, 4]) / 2.
        mid_f_2_x = (corners[0, 5] + corners[0, 1]) / 2.
        mid_f_1_y = (corners[1, 0] + corners[1, 4]) / 2.
        mid_f_2_y = (corners[1, 5] + corners[1, 1]) / 2.
        xc_f = (mid_f_1_x + mid_f_2_x) / 2.
        yc_f = (mid_f_1_y + mid_f_2_y) / 2.
        alpha_f = math.atan((mid_f_2_y - mid_f_1_y) / (mid_f_2_x - mid_f_1_x))
        w_f = mid_f_2_x - mid_f_1_x
        h_f = (corners[1, 0] + corners[1, 1] - corners[1, 4] - corners[1, 5]) / 2

        mid_b_1_x = (corners[0, 3] + corners[0, 7]) / 2.
        mid_b_2_x = (corners[0, 2] + corners[0, 6]) / 2.
        mid_b_1_y = (corners[1, 3] + corners[1, 7]) / 2.
        mid_b_2_y = (corners[1, 2] + corners[1, 6]) / 2.
        xc_b = (mid_b_1_x + mid_b_2_x) / 2.
        yc_b = (mid_b_1_y + mid_b_2_y) / 2.
        alpha_b = math.atan((mid_b_2_y - mid_b_1_y) / (mid_b_2_x - mid_b_1_x))
        w_b = mid_b_2_x - mid_b_1_x
        h_b = (corners[1, 2] + corners[1, 3] - corners[1, 6] - corners[1, 7]) / 2

    else:
        mid_f_2_x = (corners[0, 0] + corners[0, 4]) / 2.
        mid_f_1_x = (corners[0, 5] + corners[0, 1]) / 2.
        mid_f_2_y = (corners[1, 0] + corners[1, 4]) / 2.
        mid_f_1_y = (corners[1, 5] + corners[1, 1]) / 2.
        xc_f = (mid_f_1_x + mid_f_2_x) / 2.
        yc_f = (mid_f_1_y + mid_f_2_y) / 2.
        alpha_f = math.atan((mid_f_2_y - mid_f_1_y) / (mid_f_2_x - mid_f_1_x))
        w_f = mid_f_2_x - mid_f_1_x
        h_f = (corners[1, 0] + corners[1, 1] - corners[1, 4] - corners[1, 5]) / 2

        mid_b_2_x = (corners[0, 3] + corners[0, 7]) / 2.
        mid_b_1_x = (corners[0, 2] + corners[0, 6]) / 2.
        mid_b_2_y = (corners[1, 3] + corners[1, 7]) / 2.
        mid_b_1_y = (corners[1, 2] + corners[1, 6]) / 2.
        xc_b = (mid_b_1_x + mid_b_2_x) / 2.
        yc_b = (mid_b_1_y + mid_b_2_y) / 2.
        alpha_b = math.atan((mid_b_2_y - mid_b_1_y) / (mid_b_2_x - mid_b_1_x))
        w_b = mid_b_2_x - mid_b_1_x
        h_b = (corners[1, 2] + corners[1, 3] - corners[1, 6] - corners[1, 7]) / 2
    encoded_coor = np.array([xc_f, yc_f, alpha_f, w_f, h_f, xc_b, yc_b, alpha_b, w_b, h_b])
    return encoded_coor


def decode_coordinate(encoded_coor):
    mid_f_1_x = encoded_coor[0] - encoded_coor[3] / 2.
    mid_f_2_x = encoded_coor[0] + encoded_coor[3] / 2.
    mid_f_1_y = encoded_coor[1] - math.tan(encoded_coor[2]) * encoded_coor[3] / 2.
    mid_f_2_y = encoded_coor[1] + math.tan(encoded_coor[2]) * encoded_coor[3] / 2.
    x1 = mid_f_1_x
    x5 = mid_f_1_x
    x2 = mid_f_2_x
    x6 = mid_f_2_x
    y1 = mid_f_1_y + encoded_coor[4] / 2.
    y5 = mid_f_1_y - encoded_coor[4] / 2.
    y2 = mid_f_2_y + encoded_coor[4] / 2.
    y6 = mid_f_2_y - encoded_coor[4] / 2.

    mid_b_1_x = encoded_coor[5] - encoded_coor[8] / 2.
    mid_b_2_x = encoded_coor[5] + encoded_coor[8] / 2.
    mid_b_1_y = encoded_coor[6] - math.tan(encoded_coor[7]) * encoded_coor[8] / 2.
    mid_b_2_y = encoded_coor[6] + math.tan(encoded_coor[7]) * encoded_coor[8] / 2.
    x4 = mid_b_1_x
    x8 = mid_b_1_x
    x3 = mid_b_2_x
    x7 = mid_b_2_x
    y4 = mid_b_1_y + encoded_coor[9] / 2.
    y8 = mid_b_1_y - encoded_coor[9] / 2.
    y3 = mid_b_2_y + encoded_coor[9] / 2.
    y7 = mid_b_2_y - encoded_coor[9] / 2.
    decoded_coor = np.array([[x1, x2, x6, x5, x4, x3, x7, x8], [y1, y2, y6, y5, y4, y3, y7, y8]]).astype(int).reshape(
        (2, 8))

    return decoded_coor


def draw_decoded_bbox(draw, decoded_bbox):
    idx = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 4, 7, 3], [1, 5, 6, 2]]).reshape((4, 4))
    if decoded_bbox != []:
        for f in range(0, 4):
            pts = []
            for j in range(4):
                point = [int(decoded_bbox[0, idx[f, j]]) + 1, int(decoded_bbox[1, idx[f, j]]) + 1]
                pts.append(point)
            pts = np.array(pts).reshape((-1, 1, 2))
            cv2.polylines(draw, [pts], isClosed=True, thickness=2, color=[255, 0, 0])
    return draw


def decode_show_bbox():
    calib_filepaths = get_filepath(root_dir)
    i = 0
    while 0 <= i < len(calib_filepaths):
        calib_filepath = calib_filepaths[i]
        p2 = readCalibration(calib_filepath)
        objs = readLabels(calib_filepath)
        pic_path = calib_filepath.replace('calib', 'image_2').replace('txt', 'png')
        img = cv2.imread(pic_path)
        draw = img.copy()
        for obj in objs:
            corners_2d, face_idx = compute3dBBox(obj, p2)
            encoded_coor = encode_coordinate(corners_2d, float(obj[14]))
            decoded_coor = decode_coordinate(encoded_coor)
            draw = draw_decoded_bbox(draw, decoded_coor)

        cv2.putText(draw, 'ESC: Quit, U: prev picture, Any other: next picture', (2, 10), cv2.FONT_HERSHEY_PLAIN,
                    1, [0, 255, 0], 2)
        cv2.imshow(pic_path, draw)
        key = cv2.waitKey(0)
        if key == 27:
            break
        elif key == 117:
            i -= 1
            cv2.destroyAllWindows()
            continue
        else:
            cv2.destroyAllWindows()
            i += 1
            continue


def make_trainfile_v2():
    train_valid_ratio = 0.8
    vehicle_cata = ['Car', 'Van', 'Tram', 'Misc', 'Truck']
    calib_filepaths = get_filepath(root_dir)
    os.makedirs('data', exist_ok=True)
    i = 0
    calib_filepaths = tqdm(calib_filepaths)
    for calib_filepath in calib_filepaths:
        p2 = readCalibration(calib_filepath)
        objs = readLabels(calib_filepath)
        pic_path = calib_filepath.replace('calib', 'image_2').replace('txt', 'png')
        img = cv2.imread(pic_path)

        for obj in objs:
            if obj[0] not in vehicle_cata:
                continue

            x1_2d = float(obj[4])
            y1_2d = float(obj[5])
            x2_2d = float(obj[6])
            y2_2d = float(obj[7])

            if (x2_2d - x1_2d) < 20 or (y2_2d - y1_2d) < 20:
                continue

            truncated = float(obj[1])
            occluded = int(obj[2])

            if truncated > 0.5 or occluded > 2:
                continue

            cropped_img = img[int(y1_2d): int(y2_2d), int(x1_2d): int(x2_2d)]
            cropped_img = cv2.copyMakeBorder(cropped_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT,
                                             value=(127.5, 127.5, 127.5))

            corners_2d, face_idx = compute3dBBox(obj, p2)

            if corners_2d == []:
                continue

            encoded_coor = encode_coordinate(corners_2d, float(obj[14]))
            encoded_coor[0] = (encoded_coor[0] - x1_2d + 10) / cropped_img.shape[1]
            encoded_coor[1] = (encoded_coor[1] - y1_2d + 10) / cropped_img.shape[0]
            encoded_coor[3] /= cropped_img.shape[1]
            encoded_coor[4] /= cropped_img.shape[0]
            encoded_coor[5] = (encoded_coor[5] - x1_2d + 10) / cropped_img.shape[1]
            encoded_coor[6] = (encoded_coor[6] - y1_2d + 10) / cropped_img.shape[0]
            encoded_coor[8] /= cropped_img.shape[1]
            encoded_coor[9] /= cropped_img.shape[0]

            cv2.imwrite('data/%d.jpg' % i, cropped_img)
            np.savetxt('data/%d.txt' % i, encoded_coor, fmt='%.6f')

            i += 1

    train_valid_thresh = int(train_valid_ratio * i)
    with open('train.txt', 'w') as f:
        for j in range(train_valid_thresh):
            f.write('data/%d.jpg\n' % j)
    with open('valid.txt', 'w') as f:
        for j in range(train_valid_thresh, i):
            f.write('data/%d.jpg\n' % j)


if __name__ == '__main__':
    make_trainfile_v2()
