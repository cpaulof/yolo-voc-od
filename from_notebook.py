import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import cv2
import tensorflow.keras.backend as kb

from data_aug.bbox_util import draw_rect
from data_aug.data_aug import RandomRotate, RandomTranslate, RandomShear, RandomHorizontalFlip



_continue = False
model_path = r'C:\Users\Paulo\VOC-OD\model_cont50_35000'

if _continue:
    new_model = tf.keras.models.load_model(model_path)
    #new_model.trainable = True
    #for k in range(16):
     #   new_model.layers[k].trainable = False
else:
    base_model = tf.keras.applications.VGG16(include_top=False)
    #base_model.summary()
    #print(base_model.layers[17].name)
    #base_model.trainable = False
    for i in range(17):
        base_model.layers[i].trainable = False
        
    output = base_model.output
    conv0 = tf.keras.layers.Conv2D(256, 3, padding="same")(output)
    #conv0 = output
    conv1 = tf.keras.layers.Conv2D(256, 3, padding="same")(conv0)
    conv2 = tf.keras.layers.Conv2D(125, 3, padding="same")(conv1)
    new_model = tf.keras.Model(base_model.input, conv2)


anchors = [1.19, 1.99,
           2.79, 4.60,
           4.54, 8.93,  
           8.06, 5.29,
           10.33, 10.65]

anchors = np.array(anchors).reshape((5,2)) / 13
IMAGE_SHAPE = (320,320,3)
GRID_SIZE = 10
GRID_PIXEL = IMAGE_SHAPE[0] // GRID_SIZE
NUM_ANCHORS = 5
BATCH_SIZE = 6
NUM_CLASSES = 20



import xml.etree.ElementTree as ET
import pickle


annotations_dir = r'C:\Users\Paulo\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\Annotations'
classes_id = {
    'person': 0,
    'bird': 1,
    'cat': 2,
    'cow': 3,
    'dog': 4,
    'horse': 5,
    'sheep': 6,
    'aeroplane': 7,
    'bicycle': 8,
    'boat': 9,
    'bus': 10,
    'car': 11,
    'motorbike': 12,
    'train': 13,
    'bottle': 14,
    'chair': 15,
    'diningtable': 16,
    'pottedplant': 17,
    'sofa': 18,
    'tvmonitor': 19
}

CLASSES = list(classes_id.keys())

def convert_bounding_boxes(image_size, xmin, ymin, xmax, ymax):
    ''' converte (xmin ymin xmax ymax) para (xcenter, ycenter, width, height)'''
    width = xmax - xmin
    height = ymax - ymin
    xcenter = xmin + width/2
    ycenter = ymin + height/2
    return [xcenter/image_size[0], ycenter/image_size[1], width/image_size[0], height/image_size[1]]
    
def parse_voc_annotations(skip_difficult=True, save=True, force=False):
    pickle_filename = r'C:\Users\Paulo\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\annotations.pkl'
    if not force and os.path.exists(pickle_filename):
        print('loading saved annotations')
        with open(pickle_filename, 'rb') as file:
            annotations = pickle.load(file)
        file.close()
        return annotations
    filenames = os.listdir(annotations_dir)
    annotations = []
    for filename in filenames:
        element = ET.parse(os.path.join(annotations_dir, filename))
        root = element.getroot()
        image_filename = root.find('filename').text
        size = root.find('size')
        image_width = int(size.find('width').text)
        image_height = int(size.find('height').text)
        objects = []
        for obj in root.iter('object'):
            if obj.find('difficult').text=='1' and skip_difficult: continue
            class_id = classes_id[obj.find('name').text]
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            #objects.append([class_id, convert_bounding_boxes((image_width, image_height), xmin,ymin,xmax,ymax)])
            objects.append([xmin, ymin, xmax, ymax, class_id])
        annotations.append([image_filename, (image_width, image_height), objects])
        
    if save:
        with open(pickle_filename, 'wb') as file:
            pickle.dump(annotations, file)
        file.close()
    return annotations





def boxes_iou(boxes1, boxes2):
    low  = np.s_[..., :2]
    high = np.s_[..., 2:]
    
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]
    
    boxes1 = np.concatenate([boxes1[low] - boxes1[high]*0.5, boxes1[low] + boxes1[high]*0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[low] - boxes2[high]*0.5, boxes2[low] + boxes2[high]*0.5], axis=-1)
    
    left_up = np.maximum(boxes1[low], boxes2[low])
    right_down = np.minimum(boxes1[high], boxes2[high])
    intersection = np.maximum(right_down - left_up, 0.0)
    intersection_area = intersection[..., 0] * intersection[..., 1]
    union_area = boxes1_area + boxes2_area - intersection_area
    return intersection_area / union_area



def find_grid(bboxes):
    _ij = bboxes[...,:2]*GRID_SIZE
    ij = np.floor(_ij)
    offset_ij = _ij - ij
    return np.concatenate((ij, _ij), axis=-1)



def preprocess_label(targets):
    ''' recebe um vetor de shape (BATCH_SIZE, 6) e retorna um 
        vetor de shape (BATCH_SIZE, 7, 7, 125)
        
        4x13x13x125
        
        4x4
        
        entrada: Array(BATCH_SIZE, [image_index, class, center_x, center_y, width, height])
        
        '''
    indices = targets[..., 0].reshape((targets.shape[0],)).astype(np.int)
    classes = targets[..., 1].reshape((targets.shape[0],)).astype(np.int)
    bboxes = targets[..., 2:]
    
    result = np.zeros(shape=(BATCH_SIZE, GRID_SIZE, GRID_SIZE, NUM_ANCHORS, NUM_CLASSES + 5), dtype=np.float32)
    
    grid = find_grid(bboxes)
    
    
    #bboxes_anchors = np.zeros((targets.shape[0], 5, 4), dtype=np.float32)
    #bboxes_anchors[:,] = bboxes
    bboxes_anchors = bboxes.repeat(5).reshape(targets.shape[0], NUM_ANCHORS, 4)
    _anchors = np.copy(bboxes_anchors)
    _anchors[..., 2:] = anchors
    best_anchors = np.argmax(boxes_iou(bboxes_anchors, _anchors), axis=-1)
    #print(grid, best_anchors)
    #print(indices)
    k = NUM_CLASSES
    result[indices, grid[:,0].astype(np.int), grid[:,1].astype(np.int), best_anchors.astype(np.int), classes] = 1.
    result[indices, grid[:,0].astype(np.int), grid[:,1].astype(np.int), best_anchors.astype(np.int), k:k+2] = grid[:,2:]#np.log(1e-6+)
    result[indices, grid[:,0].astype(np.int), grid[:,1].astype(np.int), best_anchors.astype(np.int), k+2:k+4] = bboxes[:,2:]#/anchors[best_anchors,:]
    #result[indices, grid[:,0].astype(np.int), grid[:,1].astype(np.int), best_anchors.astype(np.int), 22:24] = np.log(bboxes[:,2:]/anchors[best_anchors,:]+1e-6)
    result[indices, grid[:,0].astype(np.int), grid[:,1].astype(np.int), best_anchors.astype(np.int), k+4] = 1
    #print(result[indices, grid[:,0].astype(np.int), grid[:,1].astype(np.int), best_anchors.astype(np.int), 22:24])
    #print(best_anchors)
    #anchors[2]
    #raise Exception
    
    return result.reshape(BATCH_SIZE, GRID_SIZE, GRID_SIZE, NUM_ANCHORS*(NUM_CLASSES + 5))



MIN_CONFIDENCE_SCORE = 0.8
MIN_CLASS_CONFIDENCE_SCORE = 0.97

#GRID_SIZE = 32.

def parse_output(output):
    output = tf.reshape(output, shape=(output.shape[0], GRID_SIZE, GRID_SIZE, NUM_ANCHORS, NUM_CLASSES + 5)).numpy()
    result = np.zeros(shape=list(output.shape[:-1])+[GRID_SIZE], dtype=np.float32)
    total = 0
    
    #output[:,:,:,24] = 
    output[:,:,:,:,NUM_CLASSES+4] = tf.nn.sigmoid(output[:,:,:,:,NUM_CLASSES + 4])
    img_idx, i_idx, j_idx, det_idx = np.where(output[:,:,:,:,NUM_CLASSES + 4] > MIN_CONFIDENCE_SCORE)
    if len(img_idx) == 0: return None
    
    # [      0, 1, 2, 3,      4,                5,          6, 7, 8, 9, 10]
    # [img_idx, i, j, d, classe, class_confidence, confidence, x, y, w, h]
    
    result = np.zeros(shape=(len(img_idx), 11))
    
    # indice da imagem e idetificação do detector (seu grid i, j e anchor box)
    result[:, 0] = img_idx
    result[:, 1] = i_idx
    result[:, 2] = j_idx
    result[:, 3] = det_idx
    
    # classe do objeto e indices de confiança
    result[:, 6] = output[img_idx, i_idx, j_idx, det_idx, NUM_CLASSES+4]
    result[:, 5] = np.max(tf.nn.softmax(output[img_idx, i_idx, j_idx, det_idx, :NUM_CLASSES]), axis=1)
    result[:, 4] = np.argmax(tf.nn.softmax(output[img_idx, i_idx, j_idx, det_idx, :NUM_CLASSES]), axis=1)
    
    
    result[:, 7]  = output[img_idx, i_idx, j_idx, det_idx, NUM_CLASSES]
    result[:, 8]  = output[img_idx, i_idx, j_idx, det_idx, NUM_CLASSES+1]
    result[:, 9]  = output[img_idx, i_idx, j_idx, det_idx, NUM_CLASSES+2]
    result[:, 10] = output[img_idx, i_idx, j_idx, det_idx, NUM_CLASSES+3]
    
    
    result[:, 5]*= result[:, 6]
    obj_idx, = np.where(result[:,5] > MIN_CLASS_CONFIDENCE_SCORE)
    
    # calculo de coordenadas x,y,w,h
    result[obj_idx, 7] = (result[obj_idx, 1] + tf.nn.sigmoid(result[obj_idx, 7]))*GRID_PIXEL
    result[obj_idx, 8] = (result[obj_idx, 2] + tf.nn.sigmoid(result[obj_idx, 8]))*GRID_PIXEL
    
    result[obj_idx, 9] =  GRID_SIZE*anchors[result[obj_idx, 3].astype(int), 0]*np.exp(result[obj_idx,  9])*GRID_PIXEL
    result[obj_idx, 10] = GRID_SIZE*anchors[result[obj_idx, 3].astype(int), 1]*np.exp(result[obj_idx, 10])*GRID_PIXEL
    #print(result[obj_idx])
    return result[obj_idx][..., (0, 4, 6, 7, 8, 9, 10)]
    



class Dataset:
    def __init__(self, batch_size=BATCH_SIZE, aug=True, aug_prob = 0.5):
        self.batch_size = batch_size
        self.image_dir = r'C:\Users\Paulo\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages'
        self.annotations = parse_voc_annotations()
        self.size = len(self.annotations)
        
        self.input_size = IMAGE_SHAPE[:2]
        
        self.batch_order = np.arange(self.size)
        np.random.shuffle(self.batch_order)
        
        self._current_id = self.batch_order[0]
         
        self.data_aug_sequence = (RandomRotate(), RandomShear(), RandomHorizontalFlip(), RandomTranslate())
        
        self.flip_coin = lambda: np.random.uniform() < (aug_prob if aug else 0.0)
    
    def gen_next_batch(self):
        images = np.zeros(shape=(self.batch_size, self.input_size[0], self.input_size[1], 3))
        batch_bboxes = np.zeros(shape=(0, 6))
        for i in range(self.batch_size):
            self._current_id = (self._current_id + i)%self.size
            filename, image_size, bboxes = self.annotations[self._current_id]
            img = np.asarray(tf.keras.preprocessing.image.load_img(os.path.join(self.image_dir, filename)))
            img, bboxes = self.apply_data_augmentation(img, bboxes)
            img, bboxes = self.resize_image(img, bboxes)
            #print(bboxes)
            bboxes = self.convert_bboxes(i, bboxes)
            #print(bboxes)
            images[i] = img
            batch_bboxes = np.concatenate((batch_bboxes, bboxes))
        
        return images/255., preprocess_label(batch_bboxes)
    
    def apply_data_augmentation(self, image, bboxes):
        _bboxes = np.array(bboxes, dtype=np.float32)
        for data_aug_func in self.data_aug_sequence:
            if self.flip_coin():
                image, _bboxes = data_aug_func(image, _bboxes)
        return image, _bboxes
    
    def resize_image(self, image, bboxes):
        
        w,h = image.shape[0], image.shape[1]
        
        image = cv2.resize(image, self.input_size)
        bboxes[...,(0,2)] /= h
        bboxes[...,(1,3)] /= w
        bboxes[..., :4]*= self.input_size[0]
        
        return image, bboxes
    
    def convert_bboxes(self, index, bboxes):
        #print('sss', bboxes)
        xywh = np.zeros((bboxes.shape[0], 4))
        xywh[..., 2] = bboxes[..., 2] - bboxes[..., 0]
        xywh[..., 3] = bboxes[..., 3] - bboxes[..., 1]
        xywh[..., 0] = bboxes[..., 0] + xywh[..., 2]*0.5
        xywh[..., 1] = bboxes[..., 1] + xywh[..., 3]*0.5
        
        bboxes[...,:4] = xywh
        bboxes[...,:4] /= self.input_size[0]
        idx = np.full((bboxes.shape[0], 1), index)
        bboxes = np.roll(bboxes, 1, axis=1)
        #print(bboxes)
        result = np.concatenate((idx, bboxes), axis=1)
        #print(result)
        return np.delete(result, np.where(result[..., 2:4]>1.0), axis=0)



OBJ_SCALE = 9.0
NO_OBJ_SCALE = 0.5
CLASS_SCALE = 1.0
COORD_SCALE = 1.0

def compute_loss(predict, target, info=False):
 
    predict = tf.reshape(predict, (BATCH_SIZE, GRID_SIZE, GRID_SIZE, NUM_ANCHORS, NUM_CLASSES + 5))
    target = tf.reshape(target, predict.shape)
    mask = tf.cast(target[..., NUM_CLASSES+4] == 1., float)
    
    _anchors = np.array(anchors)
    num_detectors = GRID_SIZE**2*NUM_ANCHORS
    num_gt = kb.sum(mask)
    num_no_gt = num_detectors*BATCH_SIZE - num_gt
    
    #class_loss = tf.nn.softmax_cross_entropy_with_logits(target[...,:NUM_CLASSES], predict[...,:NUM_CLASSES])
    _target = target[...,:NUM_CLASSES]
    _predict = predict[...,:NUM_CLASSES] 
    
    class_loss = tf.keras.backend.categorical_crossentropy(target=_target, output=_predict, from_logits=True)
    class_loss = CLASS_SCALE * tf.reduce_sum(class_loss*mask) / num_gt
    
    coord_y = tf.reshape(tf.repeat(tf.tile(tf.range(GRID_SIZE), [GRID_SIZE]), NUM_ANCHORS), (1, GRID_SIZE, GRID_SIZE, NUM_ANCHORS, 1))
    coord_y = tf.reshape(tf.repeat(coord_y, predict.shape[0], axis=0), (predict.shape[0], GRID_SIZE,GRID_SIZE,NUM_ANCHORS,1))
    coord_y = tf.cast(coord_y, tf.float32)
    coord_x = tf.transpose(coord_y, (0,2,1,3,4))
    coords = tf.concat([coord_x, coord_y], -1)
    
    pred_xy = kb.sigmoid(predict[...,NUM_CLASSES:NUM_CLASSES+2])
    pred_xy += coords
    pred_wh = kb.exp(predict[...,NUM_CLASSES+2:NUM_CLASSES+4])
    pred_wh = pred_wh*anchors
   
    coord_mask = tf.repeat(tf.reshape(mask, (BATCH_SIZE,GRID_SIZE,GRID_SIZE,NUM_ANCHORS,1)), 2, axis=-1)
    xy_loss = COORD_SCALE * kb.sum(coord_mask * kb.square(target[...,NUM_CLASSES:NUM_CLASSES+2] - pred_xy)) / (num_gt + 1e-6)
    wh_loss = COORD_SCALE * kb.sum(coord_mask * kb.square(kb.sqrt(target[...,NUM_CLASSES+2:NUM_CLASSES+4]) - kb.sqrt(pred_wh)))/(num_gt + 1e-6)
    
    coord_loss = xy_loss + wh_loss
    
    ###-----
    pred_conf = kb.sigmoid(predict[...,NUM_CLASSES+4])
    iou_wh = target[...,NUM_CLASSES+2:NUM_CLASSES+4]#*anchors
    
    
    
    xywh = tf.concat((target[...,NUM_CLASSES:NUM_CLASSES+2], iou_wh), axis=-1).numpy()
    
    
    xywh*=GRID_PIXEL
    xywh[...,2:4]*=GRID_SIZE
    xywh_pred = tf.concat((pred_xy, pred_wh), axis=-1).numpy()
    xywh_pred*=GRID_PIXEL
    xywh_pred[...,2:4]*=GRID_SIZE

    ious = boxes_iou(xywh, xywh_pred)
    
    obj_loss =  mask*(ious - tf.sigmoid(predict[...,NUM_CLASSES+4]))**2.
    obj_loss = kb.sum(obj_loss)
    obj_loss = OBJ_SCALE * obj_loss / num_gt
    
    no_object = tf.cast(ious<0.6, tf.float32)
    no_object_mask = no_object * (1. - mask)
    num_no_object = kb.sum(tf.cast(no_object_mask > 0.0, tf.float32))
    
    no_obj_loss = NO_OBJ_SCALE * kb.sum(no_object_mask*kb.square(-pred_conf)) / (num_no_object + 1e-6)

    if info:
        print('Object Loss:\t\t{}\nNo Object Loss:\t\t{}\nClass loss:\t\t{}\nCoord loss: \t\t{}\n'.format(_loss1.numpy(),
                                                                                                      _loss2.numpy(),
                                                                                                      _loss3.numpy(),
                                                                                                      _loss4.numpy()))
    return obj_loss + no_obj_loss + class_loss + coord_loss,\
            [obj_loss, no_obj_loss , class_loss , coord_loss]



dataset = Dataset(aug=True)
images, targets = dataset.gen_next_batch()
#print(images.shape, targets.shape)
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.0005)
optimizer = tf.keras.optimizers.Adagrad(learning_rate=1e-4)
#optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
print(len(new_model.trainable_variables))



history = []
import time
with tf.device('/device:GPU:0'):

    def grad(images, targets):
        #print(predict)
        with tf.GradientTape() as tape:
            predict = new_model(images, training=True)
            loss_value, loss_cat = compute_loss(predict, targets)
            #loss_value, loss_cat = comp_loss(predict, targets)
        return loss_value, loss_cat, tape.gradient(loss_value, new_model.trainable_variables)

    loss_value, loss_cat = compute_loss(new_model(images), targets)
    #loss_value, loss_cat = comp_loss(new_model(images), targets)
    print("Iteracao: {}, Loss: {}".format(optimizer.iterations.numpy(),
                                              loss_value.numpy()))
    inicio = time.time()
    total_loss = 0
    for i in range(15001):
        #if i%100 == 0:
        #    OBJ_SCALE += 1.1
        total_loss += loss_value.numpy()
        if i%250==0 and i>0:
            history.append(total_loss/250.)
            total_loss = 0.
        try:
            loss_value, loss_cat, grads = grad(images, targets)
            images, targets = dataset.gen_next_batch()
        except Exception as e:
            print(e)
            continue
        #history.append(loss_value.numpy())
        if i%1000 == 0:
            print('')
        print('\rIteracao: {}\tLoss: {:.4f} \tobj: {:.2f}\tno_obj: {:.2f}\tclass: {:.2f}\tcoord: {:.2f}'.format(i+1, loss_value.numpy(),
        
                                                                                                                *[j.numpy() for j in loss_cat]), end="")

        #optimizer.apply_gradients(zip(grads, new_model.trainable_variables))


        optimizer.apply_gradients(zip(grads, new_model.trainable_variables))
        
        if i%5000 == 0 and i>1:
            pass
            #save each 5k
            new_model.save(r'C:\Users\Paulo\VOC-OD\model_cont80_'+str(i+30000))
            

fim = time.time()
print("\ncompleto em: {:.2f} segundos".format(fim-inicio))
plt.plot(history)
plt.show()



