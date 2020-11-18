import cv2
import json
import os


with open('c:/DATASET/BOXY/boxy_labels_valid.json', 'r') as f:
    data = json.load(f)

os.makedirs('data/', exist_ok=True)

i = 0

for key in data:
    if i > 100000:
        break
    img_path = 'C:/DATASET/BOXY' + key[1:]
    img = cv2.imread(img_path)
    if not os.path.exists(img_path):
        #with open('miss_pic.txt', a) as fm:
            #fm.write(img_path + '\n')
        continue
    vehicles = data[key]['vehicles']

    print(img_path)

    for vehicle in vehicles:
        ##exclude the cars without 3d bbox
        if vehicle['AABB'] == None:
            with open('error.txt', 'a') as fe:
                fe.write('bbox wrong: %s\n' % img_path)
                continue
        else:
            x1_2d = float(vehicle['AABB']['x1']) / 2.
            y1_2d = float(vehicle['AABB']['y1']) / 2.
            x2_2d = float(vehicle['AABB']['x2']) / 2.
            y2_2d = float(vehicle['AABB']['y2']) / 2.
            # remove the too small obj
            if (x2_2d - x1_2d) < 20 or (y2_2d - y1_2d) < 20:
                continue
            if vehicle['side'] == None:
                x1 = float(vehicle['rear']['x1']) / 2.
                y1 = float(vehicle['rear']['y1']) / 2.
                x2 = float(vehicle['rear']['x2']) / 2.
                y2 = float(vehicle['rear']['y2']) / 2.
                p2x = x1_2d
                p2y = y1_2d
                p3y = y1_2d
                confidence = 0
            elif vehicle['rear'] == None:
                x1 = x1_2d
                y1 = y1_2d
                x2 = x1_2d
                y2 = y1_2d
                p2x = float(vehicle['side']['p2']['x']) / 2.
                p2y = float(vehicle['side']['p2']['y']) / 2.
                p3y = float(vehicle['side']['p3']['y']) / 2.
                confidence = 0
            else:
                x1 = float(vehicle['rear']['x1']) / 2.
                y1 = float(vehicle['rear']['y1']) / 2.
                x2 = float(vehicle['rear']['x2']) / 2.
                y2 = float(vehicle['rear']['y2']) / 2.
                p2x = float(vehicle['side']['p2']['x']) / 2.
                p2y = float(vehicle['side']['p2']['y']) / 2.
                p3y = float(vehicle['side']['p3']['y']) / 2.
                if x1<x1_2d or x2>x2_2d or y1<y1_2d or y2>y2_2d or abs(x1-x2)<5 or abs(p2x-x1)<5 or abs(p2x-x2)<5 or (p3y-p2y)<5:
                    confidence = 0
                else:
                    confidence = 1

        ##crop the cars from the original picture and add the border
        cropped_img = img[int(y1_2d):int(y2_2d), int(x1_2d):int(x2_2d)]
        cropped_img = cv2.copyMakeBorder(cropped_img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(127.5, 127.5, 127.5))

        #calculate the relative coordinate of the 3d bbox in the 2d bbox
        txc = '%.6f' % (((x1 + x2) / 2. - x1_2d + 20) / cropped_img.shape[1])
        tyc = '%.6f' % (((y1 + y2) / 2. - y1_2d + 20) / cropped_img.shape[0])
        tw = '%.6f' % ((x2 - x1) / cropped_img.shape[1])
        th = '%.6f' % ((y2 - y1) / cropped_img.shape[0])
        tp2x = '%.6f' % ((p2x - x1_2d + 20) / cropped_img.shape[1])
        tp2y = '%.6f' % ((p2y - y1_2d + 20) / cropped_img.shape[0])
        tp3y = '%.6f' % ((p3y - y1_2d + 20) / cropped_img.shape[0])
        #save the img and label file
        cv2.imwrite('data/valid_%d.jpg' % i, cropped_img)
        with open('data/valid_%d.txt' % i, 'w') as ft:
            ft.write(txc + ' ' + tyc + ' ' + tw + ' ' + th + ' ' + tp2x + ' ' + tp2y + ' ' + tp3y + ' ' + str(confidence))
        i += 1

with open('valid.txt', 'w') as f:
    for j in range(i):
        f.write('data/valid_%d.jpg\n' % j)
