import tensorflow as tf
from models.nets import cpm_hand_slim
import numpy as np
from utils import cpm_utils
import cv2
import time
import math
import sys
import random 
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout,InputLayer
import printer
"""Parameters
"""
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('DEMO_TYPE',
                           default_value='MULTI',
                           # default_value='SINGLE',
                           docstring='MULTI: show multiple stage,'
                                     'SINGLE: only last stage,'
                                     'HM: show last stage heatmap,'
                                     'paths to .jpg or .png image')
tf.app.flags.DEFINE_string('model_path',
                           default_value='models/weights/cpm_hand.pkl',
                           docstring='Your model')
tf.app.flags.DEFINE_integer('input_size',
                            default_value=368,
                            docstring='Input image size')
tf.app.flags.DEFINE_integer('hmap_size',
                            default_value=46,
                            docstring='Output heatmap size')
tf.app.flags.DEFINE_integer('cmap_radius',
                            default_value=21,
                            docstring='Center map gaussian variance')
tf.app.flags.DEFINE_integer('joints',
                            default_value=21,
                            docstring='Number of joints')
tf.app.flags.DEFINE_integer('stages',
                            default_value=4,
                            docstring='How many CPM stages')
tf.app.flags.DEFINE_integer('cam_num',
                            default_value=0,
                            docstring='Webcam device number')
tf.app.flags.DEFINE_bool('KALMAN_ON',
                         default_value=False,
                         docstring='enalbe kalman filter')
tf.app.flags.DEFINE_float('kalman_noise',
                            default_value=3e-2,
                            docstring='Kalman filter noise value')
tf.app.flags.DEFINE_string('color_channel',
                           default_value='RGB',
                           docstring='')

# Set color for each finger
joint_color_code = [[139, 53, 255],
                    [0, 56, 255],
                    [43, 140, 237],
                    [37, 168, 36],
                    [147, 147, 0],
                    [70, 17, 145]]

printer_x = 30
printer_y = 30
printer_string = ''

limbs = [[0, 1],
         [1, 2],
         [2, 3],
         [3, 4],
         [0, 5],
         [5, 6],
         [6, 7],
         [7, 8],
         [0, 9],
         [9, 10],
         [10, 11],
         [11, 12],
         [0, 13],
         [13, 14],
         [14, 15],
         [15, 16],
         [0, 17],
         [17, 18],
         [18, 19],
         [19, 20]
         ]

if sys.version_info.major == 3:
    PYTHON_VERSION = 3
else:
    PYTHON_VERSION = 2



def main(argv):
    tf_device = '/cpu:0'
    with tf.device(tf_device):
        """Build graph
        """
        if FLAGS.color_channel == 'RGB':
            input_data = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.input_size, FLAGS.input_size, 3],
                                        name='input_image')
        else:
            input_data = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.input_size, FLAGS.input_size, 1],
                                        name='input_image')

        center_map = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.input_size, FLAGS.input_size, 1],
                                    name='center_map')
        model = cpm_hand_slim.CPM_Model(FLAGS.stages, FLAGS.joints + 1)
        model.build_model(input_data, center_map, 1)
    
    saver = tf.train.Saver()

    """Create session and restore weights
    """
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    if FLAGS.model_path.endswith('pkl'):
        model.load_weights_from_file(FLAGS.model_path, sess, False)
        
    else:
        saver.restore(sess, FLAGS.model_path)

    test_center_map = cpm_utils.gaussian_img(FLAGS.input_size, FLAGS.input_size, FLAGS.input_size / 2,
                                             FLAGS.input_size / 2,
                                             FLAGS.cmap_radius)
    test_center_map = np.reshape(test_center_map, [1, FLAGS.input_size, FLAGS.input_size, 1])

    # Check weights
    for variable in tf.trainable_variables():
        with tf.variable_scope('', reuse=True):
            var = tf.get_variable(variable.name.split(':0')[0])
            print(variable.name, np.mean(sess.run(var)))

    """if not FLAGS.DEMO_TYPE.endswith(('png', 'jpg')):
        cam = cv2.VideoCapture(FLAGS.cam_num)"""

    # Create kalman filters
    if FLAGS.KALMAN_ON:
        kalman_filter_array = [cv2.KalmanFilter(4, 2) for _ in range(FLAGS.joints)]
        for _, joint_kalman_filter in enumerate(kalman_filter_array):
            joint_kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                            np.float32)
            joint_kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
            joint_kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                           np.float32) * FLAGS.kalman_noise
    else:
        kalman_filter_array = None

    #classifier MLP model
    model2 = Sequential()
    model2.add(InputLayer(input_shape=(21,2)))
    model2.add(Flatten())
    model2.add(Dense(21))
    model2.add(Dense(500))
    model2.add(Dense(1400))
    model2.add(Dense(3000))
    model2.add(Dense(3000))
    model2.add(Dense(1400))
    model2.add(Dense(500,activation='relu'))
    model2.add(Dense(26,activation='softmax'))
    model2.load_weights('models/weights/massey_joint.h5')


    notepad_image = 255 * np.ones(shape=[368, 500, 3], dtype=np.uint8)
    massey_static = cv2.imread('massey_static.png')
    massey_static = cv2.resize(massey_static,(368,368))


    with tf.device(tf_device):

        while True:
            cam = cv2.VideoCapture(FLAGS.cam_num)
   
            test_img = cpm_utils.read_image([], cam, FLAGS.input_size, 'WEBCAM')
            cam.release() 
            test_img_resize = cv2.resize(test_img, (FLAGS.input_size, FLAGS.input_size))
            t1 = time.time()
            #print('img read time %f' % (time.time() - t1))

            """if FLAGS.color_channel == 'GRAY':
                test_img_resize = np.dot(test_img_resize[..., :3], [0.299, 0.587, 0.114]).reshape(
                    (FLAGS.input_size, FLAGS.input_size, 1))
                cv2.imshow('color', test_img.astype(np.uint8))
                cv2.imshow('gray', test_img_resize.astype(np.uint8))
                cv2.waitKey(1)"""

            test_img_input = test_img_resize / 256.0 - 0.5
            test_img_input = np.expand_dims(test_img_input, axis=0)


            """if FLAGS.DEMO_TYPE.endswith(('png', 'jpg')):
                # Inference
                t1 = time.time()
                predict_heatmap, stage_heatmap_np = sess.run([model.current_heatmap,
                                                              model.stage_heatmap,
                                                              ],
                                                             feed_dict={'input_image:0': test_img_input,
                                                                        'center_map:0': test_center_map})

                # Show visualized image
               
                demo_img = visualize_result(test_img, FLAGS, stage_heatmap_np, kalman_filter_array)
                
                
                #demo_img = cv2.cvtColor(demo_img,cv2.COLOR_BGR2GRAY)
                print(demo_img.shape)
                #demo_img1 = cv2.GaussianBlur(demo_img, (9, 9), 0)
                #demo_img = 2.5*demo_img-demo_img1
                cv2.imshow('demo_img', demo_img.astype(np.uint8))
                if cv2.waitKey(0) == ord('q'): break
                print('fps: %.2f' % (1 / (time.time() - t1)))
            elif FLAGS.DEMO_TYPE == 'MULTI':"""

                # Inference
            t1 = time.time()
            #print("chandu")
            key1 = cv2.waitKey(1)
            #if key1 == 32:
            predict_heatmap, stage_heatmap_np = sess.run([model.current_heatmap,
                                                              model.stage_heatmap,
                                                              ],
                                                             feed_dict={'input_image:0': test_img_input,
                                                                        'center_map:0': test_center_map})

            # Show visualized image
              
            demo_img, joint_points= visualize_result(test_img, FLAGS, stage_heatmap_np, kalman_filter_array)
            
            
  
            x = joint_points[0][0]
            y = joint_points[0][1]
            for i in range(21):
                joint_points[i][0] =joint_points[i][0]-x
                joint_points[i][1] =joint_points[i][1]-y
            joint_points=np.array([joint_points])
         
            class_num = model2.predict_classes(joint_points)
            if class_num[0]==26:
                 character = ' '
            else:
                 character = chr(97+class_num[0])
            global printer_x,printer_y,printer_string
            printer_string+=character
            
            notepad_image = printer.notepad(notepad_image,(printer_x,printer_y),printer_string)
            new_note= np.concatenate((notepad_image,demo_img,massey_static),axis =1)
            cv2.imshow('notepad',new_note)
            if len(printer_string)==17:
                 printer_string = ''
                 printer_y+=25
            print('fps: %.2f' % ( 1/(time.time() - t1)))
            
            if key1 == ord('q'):
                cv2.destroyAllWindows()
                break
            #print('fps: %.2f' % (1 / (time.time() - t1)))


            


def visualize_result(test_img, FLAGS, stage_heatmap_np, kalman_filter_array):
    
    t1 = time.time()
    demo_stage_heatmaps = []
    joint_points = []
    if FLAGS.DEMO_TYPE == 'MULTI' or FLAGS.DEMO_TYPE.endswith(('png', 'jpg')):
        for stage in range(len(stage_heatmap_np)):
            demo_stage_heatmap = stage_heatmap_np[stage][0, :, :, 0:FLAGS.joints].reshape(
                (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
            demo_stage_heatmap = cv2.resize(demo_stage_heatmap, (test_img.shape[1], test_img.shape[0]))
            demo_stage_heatmap = np.amax(demo_stage_heatmap, axis=2)
            demo_stage_heatmap = np.reshape(demo_stage_heatmap, (test_img.shape[1], test_img.shape[0], 1))
            demo_stage_heatmap = np.repeat(demo_stage_heatmap, 3, axis=2)
            demo_stage_heatmap *= 255
            demo_stage_heatmaps.append(demo_stage_heatmap)

        last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.joints].reshape(
            (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
        last_heatmap = cv2.resize(last_heatmap, (test_img.shape[1], test_img.shape[0]))
    else:
        last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.joints].reshape(
            (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
        last_heatmap = cv2.resize(last_heatmap, (test_img.shape[1], test_img.shape[0]))
    #print('hm resize time %f' % (time.time() - t1))

    t1 = time.time()
    joint_coord_set = np.zeros((FLAGS.joints, 2))

    # Plot joint colors
    if kalman_filter_array is not None:
        for joint_num in range(FLAGS.joints):
            joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                           (test_img.shape[0], test_img.shape[1]))
            joint_coord = np.array(joint_coord).reshape((2, 1)).astype(np.float32)
            kalman_filter_array[joint_num].correct(joint_coord)
            kalman_pred = kalman_filter_array[joint_num].predict()
            joint_coord_set[joint_num, :] = np.array([kalman_pred[0], kalman_pred[1]]).reshape((2))

            color_code_num = (joint_num // 4)
            if joint_num in [0, 4, 8, 12, 16]:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
            else:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
    else:
        for joint_num in range(FLAGS.joints):
            joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                           (test_img.shape[0], test_img.shape[1]))
            joint_coord_set[joint_num, :] = [joint_coord[0], joint_coord[1]]

            color_code_num = (joint_num // 4)
            if joint_num in [0, 4, 8, 12, 16]:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
            else:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
            joint_points.append([joint_coord[1], joint_coord[0]])
   # print('plot joint time %f' % (time.time() - t1))

    t1 = time.time()
    # Plot limb colors
    for limb_num in range(len(limbs)):
        
        x1 = joint_coord_set[limbs[limb_num][0], 0]
        y1 = joint_coord_set[limbs[limb_num][0], 1]
        x2 = joint_coord_set[limbs[limb_num][1], 0]
        y2 = joint_coord_set[limbs[limb_num][1], 1]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if length < 150 :
            deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
            polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                       (int(length / 2), 3),
                                       int(deg),
                                       0, 360, 1)
            color_code_num = limb_num // 4
            if PYTHON_VERSION == 3:
                limb_color = list(map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num]))
            else:
                limb_color = map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num])

            cv2.fillConvexPoly(test_img, polygon, color=limb_color)
    #print('plot limb time %f' % (time.time() - t1))

    if FLAGS.DEMO_TYPE == 'MULTI':
        #upper_img = np.concatenate((demo_stage_heatmaps[0], demo_stage_heatmaps[1], demo_stage_heatmaps[2]), axis=1)
        #lower_img = np.concatenate((demo_stage_heatmaps[3], demo_stage_heatmaps[len(stage_heatmap_np) - 1], test_img),axis=1)
        #demo_img = np.concatenate((upper_img, lower_img), axis=0)
        return  test_img,joint_points
    else:
        return  test_img,joint_points

if __name__ == '__main__':
    tf.app.run()
