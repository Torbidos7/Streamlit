import numpy as np
import cv2
import pandas as pd
# import matplotlib.pyplot as plt
from glob import glob
from collections import Counter
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
from segmentation_models import Unet
import tensorflow as tf
import argparse
import textwrap

###########################################################################################
#
# Bulb detection script given a RGB dermoscopic image
# Given a dermoscopic image, the script detects the hair bulbs and the direction of the hair
#
###########################################################################################

#------------------------------------------------------------------------------------------

#constants
BULB_DEPTH = 1.5 #mm

#constants of the model
BATCH = 8
IMG_SIZE = 224

# versbose print, substitute for print when argue verbose is True
verboseprint =  lambda *a, **k: None

#enviroment variable
sm.set_framework('tf.keras')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#pats
weights_file='models/mobilenetv2_tversky_segmentation_models.h5'


def find_line(p1, p2):
    '''
    Find the line that passes through two points

    Args:
        p1 (tuple): first point
        p2 (tuple): second point

    Returns:
        m (float): slope of the line
        b (float): intercept of the line
    '''
    try:
        m = (p2[1]-p1[1])/(p2[0]-p1[0])
        b = p1[1]-m*p1[0]
      
    except:
      
        verboseprint('vertical line')
    return m, b


def distance(p1, p2):
    '''
    Find the distance between two points

    Args:
        p1 (tuple): first point
        p2 (tuple): second point

    Returns:
        distance (float): distance between the two points
    '''
    return np.sqrt((p2[1]-p1[1])**2+(p2[0]-p1[0])**2)


def image_resize(img):
    '''
    Resize image to 224x224 for model prediction and normalize it
    
    Parameters: 
        img (numpy array): image to be resized
        
    Returns:
        img (numpy array): resized image
    '''
    img = tf.image.resize(img, size=(IMG_SIZE, IMG_SIZE), preserve_aspect_ratio=False)
    img *= 1/255.
    img = tf.cast(img, tf.float32)

    return img


###########################################################################################
#
# Segmentation part of the script, given a dermoscopic image it returns the mask of the hair
#
###########################################################################################

def detect_bulb(frame):
    # load model
    model = Unet('mobilenetv2', input_shape=(IMG_SIZE, IMG_SIZE, 3), encoder_weights='imagenet', classes=1, activation='sigmoid')
    model.compile(optimizer='adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])
    model.load_weights(weights_file)

    images= []
    dimensions = []
    img = frame
    dimensions.append(img.shape[:2])
    img = image_resize(img)    
    images.append(img)

    verboseprint('Images: ', len(images))

    # Predict masks
    pred_masks = model.predict(tf.convert_to_tensor(images))

    # Resize masks to original size and binarize
    masks = [cv2.resize(pr_mask, dsize=(dim[1], dim[0]), interpolation=cv2.INTER_NEAREST) for pr_mask, dim in zip(pred_masks, dimensions)]
    masks = [np.where(mask[...] > .5, 255, 0).astype('uint8') for mask in masks]



    ###########################################################################################
    #
    #Bulb detection part of the script, given the mask of the hair it returns the coordinates of the bulb
    # and the direction of the hair 
    #
    ############################################################################################

    
    #read image and mask
    img = frame
    mask = masks[0]
    

    #container for the outern point of each hair
    right_points_total = []
    max_box_total = []
    count_total = []

    #connect components
    connectivity = 8
    connected_components = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
    # The first cell is the number of labels
    num_labels = connected_components[0]
    # The second cell is the label matrix
    labels = connected_components[1]
    # The third cell is the stat matrix
    stats = connected_components[2]
    # The fourth cell is the centroid matrix
    centroids = connected_components[3]

    #container for the bounding boxes and the masks of each hair
    bounding_boxes = [] 
    componentMasks = []
    
    #loop over the connected components
    for i in range(num_labels):
        x,y,w,h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2, cv2.LINE_AA, ) #draw bounding box
        # check if the area is too small
        if w*h>50:
            #save bounding box and mask
            bounding_boxes.append((x,y,w,h))
            componentMasks.append(np.expand_dims((labels == i).astype("uint8"), axis=-1))

    #convert to numpy array
    bounding_boxes = np.array(bounding_boxes)
    componentMasks = np.array(componentMasks)


    #work on single bounding box
    for c_num, _ in enumerate(bounding_boxes):

        #get single hair coordinates
        x,y,w,h = bounding_boxes[c_num]

        #get single hair mask and original image
        single_hair_original = img[ y:y+h, x:x+w]
        single_hair_mask = componentMasks[c_num][ y:y+h, x:x+w]
        single_hair= single_hair_original*single_hair_mask


        #create 4 bounding box of increasing size around the corners of the image
        box_dim=0

        #get max x and y
        max_x= single_hair.shape[0]
        max_y= single_hair.shape[1]

        #count number of corners with more than 0 pixels lightened up    
        non_zero=0

        #increase box size until there are at least 2 corners with more than 0 pixels lightened up
        while (non_zero<2):

            #increase box size
            box_dim+=1 
            
            #top left
            tl = single_hair[0:box_dim, 0:box_dim ]
            tl_original = single_hair_original[0:box_dim, 0:box_dim]
            tl_mask = single_hair_mask[0:box_dim, 0:box_dim]

            #top right
            tr = single_hair[0:box_dim, max_y-box_dim:max_y ]
            tr_original = single_hair_original[0:box_dim, max_y-box_dim:max_y]
            tr_mask = single_hair_mask[0:box_dim, max_y-box_dim:max_y]

            #bottom left
            bl = single_hair[max_x-box_dim:max_x, 0:box_dim]
            bl_original = single_hair_original[max_x-box_dim:max_x, 0:box_dim]
            bl_mask = single_hair_mask[max_x-box_dim:max_x, 0:box_dim]

            #bottom right
            br = single_hair[max_x-box_dim:max_x, max_y-box_dim:max_y]
            br_original = single_hair_original[max_x-box_dim:max_x, max_y-box_dim:max_y]
            br_mask = single_hair_mask[max_x-box_dim:max_x, max_y-box_dim:max_y]

            #non zero increase if there is one corner with more than 0 pixels
            count = [1 if( np.count_nonzero(tl)) else 0, 1 if( np.count_nonzero(tr)) else 0, 1 if( np.count_nonzero(bl)) else 0, 1 if( np.count_nonzero(br))else 0] 
            non_zero = sum(count)



        #based on count index draw bounding box around the corner with the most pixels
        box_dim=max(box_dim, (min(max_x, max_y)//3))

        tl = single_hair[0:box_dim, 0:box_dim ]
        tr = single_hair[0:box_dim, max_y-box_dim:max_y ]
        bl = single_hair[max_x-box_dim:max_x, 0:box_dim]
        br = single_hair[max_x-box_dim:max_x, max_y-box_dim:max_y]

        #create list of shifts for each corner to come back to the original image
        shifts = [[0,0], [0, max_y-box_dim], [max_x-box_dim, 0], [max_x-box_dim, max_y-box_dim]]
        max_box = []
        points = []

        #loop over the corners
        boxes = [tl, tr, bl, br]

        for box, c  in zip ( boxes, count):
            #check if there are pixels in the corner
            if c:
                #cv2.rectangle(single_hair,(0,0),(box_dim,box_dim),(0,255,0),2 )
                #get max value in the corner
                max_box.append((np.max(box)))
                #get coordinates of the max value           
                points.append(np.squeeze(np.argwhere(box == np.max(box))[0]))

            else:
                #if there are no pixels in the corner append 0
                points.append([0,0, 2])
                max_box.append(0)
                continue

        #shift points to the original image
        if count[0] :
            points[0]+=np.array([0,0,0])
            points[0] = points[0].tolist()
        if count[1] :
            points[1]+=np.array([0,max_y-box_dim,0])
            points[1] = points[1].tolist()
        if count[2] :    
            points[2]+=np.array([max_x-box_dim,0,0])
            points[2] = points[2].tolist()
        if count[3] :   
            points[3]+=np.array([max_x-box_dim,max_y-box_dim,0])
            points[3] = points[3].tolist()


        # verboseprint(count , max_box, points)


        #select right points
        rigth_points = np.array(points)
        count = np.array(count)
        max_box = np.array(max_box)


        #select only points where count is 1
        max_box = max_box[count==1]
        rigth_points = rigth_points[count==1]
        
    
        #correct coordinates for entire image ################ ONLY NEEDED FOR FULL IMAGE PREDICTION
        rigth_points[:,0] += y
        rigth_points[:,1] += x

        right_points_total.append(rigth_points)
        max_box_total.append(max_box)
        count_total.append(count)
        
    #SAVE FULL IMAGE WITH HAIR DIRECTIONS    


    calibration_factor = 75. #pixels per mm

    #majority vote
    majority = []
    
    for maxi in max_box_total:
        if maxi[0]>maxi[1]:
            majority.append(0)
        else:
            majority.append(1)
            
        count = Counter(majority)
        count.most_common(1)[0][0]

    #correction to right_points_total
    #keep only 2 couples of points if there are more than 2 couples only if the coordinate x or y are equal



    wrong_points_total = [couple[:,:2] for couple in right_points_total if couple.shape[0]!=2]
    right_points_total = [couple[:,:2] for couple in right_points_total if couple.shape[0]==2]

    #find m values for each couple of points (hair direction)
    linear_coefficients = [find_line(couple[0][::-1], couple[1][::-1])[0] for couple in right_points_total]
    intercepts = [find_line(couple[0][::-1], couple[1][::-1])[1] for couple in right_points_total]  
    
    #create dataframe for bulb points
    bulb_points = pd.DataFrame(columns=['x', 'y'])

    #verboseprint IMAGE WITH HAIR DIRECTIONS BASED ON MAJORITY VOTE 
    # plt.imshow(img) 
    if count.most_common(1)[0][0]:
        title ="Hair is pointing down"
        for couple, count, m , b in zip(right_points_total, count_total, linear_coefficients, intercepts):

            
            # plt.scatter(couple[0,1], couple[0,0], marker="x", color="yellow", s=200) # [0,1] is x, [0,0] is y
            # plt.scatter(couple[1,1], couple[1,0], marker="x", color="green", s=200) #
                    
            
            x_bulb =  couple[0,1]
            if m < 0:
                x_bulb += np.abs(int((BULB_DEPTH*calibration_factor/distance(couple[0], couple[1]))*np.abs(couple[1,1]-couple[0,1]))) 
            elif m > 0:
                x_bulb -= np.abs(int((BULB_DEPTH*calibration_factor/distance(couple[0], couple[1]))*np.abs(couple[1,1]-couple[0,1])))
            # y_bulb = couple[0,0]
            y_bulb = (m*x_bulb) + b
            
            len_out = distance(couple[0], couple[1])/calibration_factor
        
            #verboseprint(len_out, m,  b, count)

            #if it falls inside the image
            if (0 < x_bulb < 1280 and 0 < y_bulb < 960):
                #plot bulb position
                # plt.scatter(x_bulb, y_bulb, marker="x", color="red", s=200)
                #add point to the dataframe
                bulb_points = bulb_points.append({'x': x_bulb, 'y': y_bulb}, ignore_index=True)
                
    
    else:
        
        title="Hair is pointing up"
        for couple, count, m , b in zip(right_points_total, count_total, linear_coefficients, intercepts):
    
            
            #chamge b for other side of the line
            b = find_line(couple[1][::-1], couple[0][::-1])[1]

            x_bulb =  couple[1,1]
            if m < 0:
                x_bulb -= np.abs(int((BULB_DEPTH*calibration_factor/distance(couple[0], couple[1]))*np.abs(couple[1,1]-couple[0,1]))) 
            elif m > 0:
                x_bulb += np.abs(int((BULB_DEPTH*calibration_factor/distance(couple[0], couple[1]))*np.abs(couple[1,1]-couple[0,1])))
            # y_bulb = couple[0,0]
            y_bulb = (m*x_bulb) + b
            
            len_out = distance(couple[0], couple[1])/calibration_factor
        
            #verboseprint(len_out, m,  b, count)

            #if it falls inside the image
            if (0 < x_bulb < 1280 and 0 < y_bulb < 960):
               
                #add point to the dataframe
                bulb_points = bulb_points.append({'x': x_bulb, 'y': y_bulb}, ignore_index=True)
          

    

    total_hair = len(wrong_points_total)+ len(right_points_total)
    percentage_bulb = len(right_points_total)/(total_hair)*100

    
    return bulb_points, mask, total_hair, percentage_bulb

    
    verboseprint(f'Detected {len(wrong_points_total)+ len(right_points_total)} hair, percentage of bulb found {len(right_points_total)/(len(wrong_points_total)+len(right_points_total))*100}%'.format(len(wrong_points_total)), 'non skipped {} contours'.format(len(right_points_total)))
