
import tensorflow as tf
import cv2
import numpy as np
import tensorflow as tf
model = tf.saved_model.load("D:/ObjectDetection_TensorFlow2/my_model_center_mobilenet_6/saved_model")


detect_fn = model.signatures['serving_default']

# detections
import json
def load_labels_map(label_url):
    # read JSON file
    a =  open(label_url)
    data = json.load(a)
    if data:
        return data['labels']
    return []

label_maps = load_labels_map('label_map_coner.pbtxt')

def get_center_point(coordinate_dict):
    """
    convert (xmin, ymin, xmax, ymax) to (x_center, y_center)
    Parameters:
        coordinate_dict (dict): dictionary of coordinates
    Returns:
        points (dict): dictionary of coordinates
    """

    points = dict()
    for key in coordinate_dict.keys():
        xmin, ymin, xmax, ymax = coordinate_dict[key][0]
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        points[key] = (x_center, y_center)
    return points

def find_miss_corner(coordinate_dict):
    """
    find the missed corner of a 
    
    Parameters:
        coordinate_dict (dict): dictionary of coordinates
    Returns:
        corner (str): name of the missed corner
    """

    dict_corner = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    for key in dict_corner:
        if key not in coordinate_dict.keys():
            return key

def calculate_missed_coord_corner(coordinate_dict):
    """
    calculate the missed coordinate of a corner
    Parameters:
        coordinate_dict (dict): dictionary of coordinates
    Returns:
        coordinate_dict (dict): dictionary of coordinates
    """
    #calulate a coord corner of a rectangle 
    def calculate_coord_by_mid_point(coor1, coord2, coord3):
        midpoint = np.add(coordinate_dict[coor1], coordinate_dict[coord2]) / 2
        y = 2 * midpoint[1] - coordinate_dict[coord3][1]
        x = 2 * midpoint[0] - coordinate_dict[coord3][0]
        return (x, y)
    # calculate missed corner coordinate
    corner = find_miss_corner(coordinate_dict)
    if corner == 'top_left':
        coordinate_dict['top_left'] = calculate_coord_by_mid_point('top_right', 
        'bottom_left', 'bottom_right')
    elif corner == 'top_right':
        coordinate_dict['top_right'] = calculate_coord_by_mid_point('top_left', 
        'bottom_right', 'bottom_left')
    elif corner == 'bottom_left':
        coordinate_dict['bottom_left'] = calculate_coord_by_mid_point('top_left', 
        'bottom_right', 'top_right')
    elif corner == 'bottom_right':
        coordinate_dict['bottom_right'] = calculate_coord_by_mid_point('bottom_left', 
        'top_right', 'top_left')
    return coordinate_dict

def perspective_transform(image, source_points):
    """
    perspective transform image
    Parameters:
        image (numpy array): base image
        source_points (numpy array): points of the image after detecting the corners
    Returns:
        image (numpy array): transformed image
    """

    # define the destination points (the points where the image will be mapped to)
    dest_points = np.float32([[0, 0], [500, 0], [500, 300], [0, 300]])

    M = cv2.getPerspectiveTransform(source_points, dest_points)
    dst = cv2.warpPerspective(image, M, (500, 300))
    return dst

def align_image(image, coordinate_dict):
    """
    align image (find the missed corner and perspective transform)
    Parameters:
        image (numpy array): image
        coordinate_dict (dict): dictionary of coordinates
    Returns:
        image (numpy array): image
    """

    if len(coordinate_dict) < 3:
        raise ValueError('Please try again')
    # convert (xmin, ymin, xmax, ymax) to (x_center, y_center)
    coordinate_dict = get_center_point(coordinate_dict)
    if len(coordinate_dict) == 3:
        coordinate_dict = calculate_missed_coord_corner(coordinate_dict)
    top_left_point = coordinate_dict['top_left']
    top_right_point = coordinate_dict['top_right']
    bottom_right_point = coordinate_dict['bottom_right']
    bottom_left_point = coordinate_dict['bottom_left']
    source_points = np.float32([top_left_point, top_right_point, bottom_right_point, bottom_left_point])
    # transform image and crop 
    crop = perspective_transform(image, source_points)
    return crop

def process_output(typee, data, threshold, targetSize, labels_map):
    """
    process output of model
    Parameters:
        type (str): type of model
        data (numpy array): output of model
        threshold (float): threshold of model
        targetSize (dict): target size of image
    Returns:
        result (dict): list bounding boxes
    """

    scores, boxes, classes = None, None, None
    label_map = None
    if typee == 'corner':
#         a = data['detection_scores']
        # b = tf.make_ndarray(a)
        scores = list(data['detection_scores'][0])
        boxes = list(data['detection_boxes'][0])
        classes = list(data['detection_classes'][0])
        label_map = labels_map
    if typee == 'text':
        scores = list(data['detection_scores'][0])
        boxes = list(data['detection_boxes'][0])
        classes = list(data['detection_classes'][0])
        label_map = labels_map
    results = {}
    for i in range(len(scores)):
        if scores[i] > threshold:
            label = label_map[int(classes[i]) - 1]
            if label in results:
                results[label].append([
                    boxes[i][1] * targetSize['w'], 
                    boxes[i][0] * targetSize['h'], 
                    boxes[i][3] * targetSize['w'], 
                    boxes[i][2] * targetSize['h']])
            else:
                results[label] = [[
                    boxes[i][1] * targetSize['w'], 
                    boxes[i][0] * targetSize['h'], 
                    boxes[i][3] * targetSize['w'], 
                    boxes[i][2] * targetSize['h']]]
    
    return results

img = cv2.imread('D:/ObjectDetection_TensorFlow2/an1.jpg')
img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
input_tensor = tf.convert_to_tensor(img)
input_tensor = input_tensor[tf.newaxis, ...]

results = detect_fn(input_tensor)

targetSize = { 'w': 0, 'h': 0 }
targetSize['h'] = img.shape[0]
targetSize['w'] = img.shape[1]

output = process_output('corner', results, 0.4, targetSize, label_maps)
crop_img = align_image(img, output)
crop_img = np.array(crop_img)
cv2.imwrite('test4.jpg', crop_img)

######################################################################################################
import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['cnn']['pretrained']=False
config['device'] = 'cpu'
config['predictor']['beamsearch']=False

detector = Predictor(config)

#######################################################################################################

model2 = tf.saved_model.load("D:/ObjectDetection_TensorFlow2/my_center_resnet101_6_1/saved_model")
label_maps = load_labels_map('label_map_1.pbtxt')

image_path = "test4.jpg"

detect_fn = model2.signatures['serving_default']
img = cv2.imread(image_path)
img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
input_tensor = tf.convert_to_tensor(img)
input_tensor = input_tensor[tf.newaxis, ...]
results = detect_fn(input_tensor)


targetSize = { 'w': 0, 'h': 0 }
targetSize['h'] = img.shape[0]
targetSize['w'] = img.shape[1]

output = process_output('text', results, 0.5, targetSize, label_maps)
img = cv2.imread(image_path)
results_str = ""


#####id
arr = np.array(output['id'])
ima = img[int(arr[0][1]):int(arr[0][3]),int(arr[0][0]):int(arr[0][2])]
cv2.imwrite('test_id.jpg', ima)
ima = 'test_id.jpg'
ima = Image.open(ima)
s = "id: " + detector.predict(ima) 
results_str = results_str + s + "\n"


#####name
arr = np.array(output['name'])
arr = sorted(arr, key=lambda x:[x[0]])
arr = np.array(arr)
leng = arr.shape[0]
s1 = "name: "
for i in range (0,leng) :
    ima = img[int(arr[i][1]):int(arr[i][3]),int(arr[i][0]):int(arr[i][2])]
    cv2.imwrite('test_name' + str(i+1) + '.jpg' , ima)
    path = 'test_name' + str(i+1) + '.jpg'
    ima = Image.open(path)
    s = detector.predict(ima) 
    s1 = s1 + s + " "
results_str = results_str + s1 + "\n"

#####dob
arr = np.array(output['dob'])
ima = img[int(arr[0][1]):int(arr[0][3]),int(arr[0][0]):int(arr[0][2])]
cv2.imwrite('test_dob.jpg', ima)
ima = 'test_dob.jpg'
ima = Image.open(ima)
s = "dob: " + detector.predict(ima) 
results_str = results_str + s + "\n"

#####sex
arr = np.array(output['sex'])
ima = img[int(arr[0][1]):int(arr[0][3]),int(arr[0][0]):int(arr[0][2])]
cv2.imwrite('test_sex.jpg', ima)
ima = 'test_sex.jpg'
ima = Image.open(ima)
s = "sex: " + detector.predict(ima) 
results_str = results_str + s + "\n"

####date
arr = np.array(output['date'])
ima = img[int(arr[0][1]):int(arr[0][3]),int(arr[0][0]):int(arr[0][2])]
cv2.imwrite('test_date.jpg', ima)
ima = 'test_date.jpg'
ima = Image.open(ima)
s = "date: " + detector.predict(ima) 
results_str = results_str + s + "\n"

####nati
arr = np.array(output['nati'])
ima = img[int(arr[0][1]):int(arr[0][3]),int(arr[0][0]):int(arr[0][2])]
cv2.imwrite('test_nati.jpg', ima)
ima = 'test_nati.jpg'
ima = Image.open(ima)
s = "nati: " + detector.predict(ima) 
results_str = results_str + s + "\n"

#####res
arr = np.array(output['pla'])
arr = sorted(arr, key=lambda x:[x[0]])
arr = np.array(arr)
leng = arr.shape[0]
s1 = "pla: "
for i in range (0,leng) :
    ima = img[int(arr[i][1]):int(arr[i][3]),int(arr[i][0]):int(arr[i][2])]
    cv2.imwrite('test_pla' + str(i+1) + '.jpg' , ima)
    path = 'test_pla' + str(i+1) + '.jpg'
    ima = Image.open(path)
    s = detector.predict(ima) 
    s1 = s1 + s + " "
results_str = results_str + s1 + "\n"


####res
arr = np.array(output['res'])
leng = arr.shape[0]
arr = sorted(arr, key=lambda x:[x[1]])
arr = np.array(arr)
Max = arr[leng-1][1]
boxes = []
i = 0
count = 0
s1 = "res: "
while i < leng - count :
    if(Max - arr[i][1] >= 4) :
        boxes.append([arr[i][0],arr[i][1],arr[i][2],arr[i][3]])
        arr = np.delete(arr, (i), axis=0)
        count += 1
    i += 1

boxes = np.array(boxes)
leng = boxes.shape[0]    

for i in range (0,leng) :
    ima = img[int(boxes[i][1]):int(boxes[i][3]),int(boxes[i][0]):int(boxes[i][2])]
    cv2.imwrite('test_res' + str(i+1) + '.jpg' , ima)
    path = 'test_res' + str(i+1) + '.jpg'
    ima = Image.open(path)
    s = detector.predict(ima) 
    s1 = s1 + s + " "

arr = sorted(arr, key=lambda x:[x[0]])
arr = np.array(arr)
i = leng
leng = arr.shape[0]

for j in range (0,leng) :
    ima = img[int(arr[j][1]):int(arr[j][3]),int(arr[j][0]):int(arr[j][2])]
    cv2.imwrite('test_res' + str(j+i+1) + '.jpg' , ima)
    path = 'test_res' + str(j+i+1) + '.jpg'
    ima = Image.open(path)
    s = detector.predict(ima) 
    s1 = s1 + s + " "
results_str = results_str + s1 + "\n"

print(results_str)














