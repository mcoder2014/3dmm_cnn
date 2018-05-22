#-*- coding: utf-8 -*-
"""
Author: Chaoqun
Time: 2018-05-21

"""

import os
import sys

sys.path.append("/home/chaoqun/github/caffe/python")
sys.path.append("/home/chaoqun/github/caffe/python/caffe")

## Tu suppress the noise output of Caffe when loading a model
## polish the output (see http://stackoverflow.com/questions/29788075/setting-glog-minloglevel-1-to-prevent-output-in-shell-from-caffe)
os.environ['GLOG_minloglevel'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]='0'
###################
import numpy as np
from PIL import Image
from glob import glob
import caffe
import cv2
import time
import ntpath
import os.path
import scipy.io
import shutil
from skimage import io
import dlib
import utils
import operator
import argparse

# 全局变量
FLAGS = None
layer_name = 'fc_ftnew'
GPU_ID = 0	                 # 使用的GPU
## CNN template size
trg_size = 224
needCrop = 1                #  tells the demo if the images need cropping (1) or not (0).
# Default 1. If your input image size is equal (square) and has a CASIA-like [2] bounding box,
#  you can set <needCrop> as 0. Otherwise, you have to set it as 1.
useLM    = 0                #  is an option to refine the bounding box
                            # using detected landmarks (1) or not (0). Default 1.

def prepareImage(imagePath):
    """ Prepare images before use it.
    """
    print("> Prepare image "+imagePath + ":")
    imname = ntpath.basename(imagePath)
    imname = imname.split(imname.split('.')[-1])[0][0:-1]       # 记录图片basename
    img = cv2.imread( imagePath )   # 加载图片

    if needCrop:
        dlib_img = io.imread(imagePath)
        img2 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(FLAGS.predictor_path)  # 加载 dlib 检测
        dets = detector(img, 1)
        print(">     Number of faces detected: {}".format(len(dets)))
        if len(dets) == 0:
            print '> Could not detect the face, skipping the image...' + image_path
            return None
        if len(dets) > 1:
            print "> Process only the first detected face!"
        # 标记第一个人脸
        detected_face = dets[0]
        cv2.rectangle(img2, (detected_face.left(),detected_face.top()), \
            (detected_face.right(),detected_face.bottom()), (0,0,255),2)
        fileout = open(os.path.join(FLAGS.tmp_detect , imname + ".bbox"),"w")
        fileout.write("%f %f %f %f\n" % (detected_face.left(),detected_face.top(), \
            detected_face.right(),detected_face.bottom()))
        fileout.close()

        ## If we are using landmarks to crop
        if useLM:
            shape = predictor(dlib_img, detected_face)
            nLM = shape.num_parts
            fileout = open(os.path.join(FLAGS.tmp_detect, imname + ".pts" ), "w")
            for i in range(0, nLM):
                cv2.circle( img2, (shape.part(i).x, shape.part(i).y), 5, (255,0,0))
                fileout.write("%f %f\n" % (shape.part(i).x, shape.part(i).y))
            fileout.close()
            img = utils.cropByLM(img, shape, img2)
        else:
            print "> cropByFaceDet "
            img = utils.cropByFaceDet(img, detected_face, img2)
        cv2.imwrite(os.path.join( FLAGS.tmp_detect, imname+"_detect.png"), img2)
    img = cv2.resize(img, (trg_size, trg_size))
    cv2.imwrite(os.path.join(FLAGS.tmp_ims, imname + '.png'), img)
    return img

def cnn_3dmm( img, outputPath ):
    """
    use 3dmm cnn to deal with
    """
    # load net
    try:
        caffe.set_mode_gpu()
        caffe.set_device(GPU_ID)
    except Exception as ex:
        print '> Could not setup Caffe in GPU ' +str(GPU_ID) + ' - Error: ' + ex
        print '> Reverting into CPU mode'
        caffe.set_mode_cpu()

    ## Opening mean average image
    proto_data = open(FLAGS.mean_path, "rb").read()
    a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
    mean  = caffe.io.blobproto_to_array(a)[0]
    ## Loading the CNN
    net = caffe.Classifier(FLAGS.deploy_path, FLAGS.model_path)

    ## Setting up the right transformer for an input image
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    transformer.set_mean('data',mean)
    print '> CNN Model loaded to regress 3D Shape and Texture!'
    ## Loading the Basel Face Model to write the 3D output
    model = scipy.io.loadmat(FLAGS.BFM_path,squeeze_me=True,struct_as_record=False)
    model = model["BFM"]
    faces = model.faces-1
    print '> Loaded the Basel Face Model to write the 3D output!'

    net.blobs['data'].reshape(1,3,trg_size,trg_size)
   # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   # im = img / 255
    imname = ntpath.basename(FLAGS.imagePath)
    imname = imname.split(imname.split('.')[-1])[0][0:-1]       # 记录图片basename

    im = caffe.io.load_image(os.path.join(FLAGS.tmp_ims, imname + '.png'))

    ## Transforming the image into the right format
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    ## Forward pass into the CNN
    net_output = net.forward()
    ## Getting the output
    features = np.hstack( [net.blobs[layer_name].data[0].flatten()] )

    #imname = ntpath.basename(FLAGS.imagePath)
    #imname = imname.split(imname.split('.')[-1])[0][0:-1]       # 记录图片basename

    ## Writing the regressed 3DMM parameters
    np.savetxt(os.path.join( outputPath, imname+ '.ply.alpha'), features[0:99])
    np.savetxt(os.path.join( outputPath, imname + '.ply.beta'), features[99:198])

    #################################
    ## Mapping back the regressed 3DMM into the original
    ## Basel Face Model (Shape)
    ##################################
    S,T = utils.projectBackBFM(model, features)
    print '> Writing 3D file in: ', os.path.join( outputPath, imname+ '.ply')
    utils.write_ply(os.path.join( outputPath, imname+ '.ply'), S, T, faces)

def main():
    """ main function
    """

    # 检查图片文件夹是否存在
    if not os.path.exists(FLAGS.imagePath):
        print("Floder {0} seems to be not existed.".format(FLAGS.path))
        return

    # 检查输出文件夹是否存在
    if not os.path.exists(FLAGS.savePath):
        os.makedirs(FLAGS.savePath)

    # 检查临时文件夹
    if not os.path.exists(FLAGS.tmp_ims):
        os.makedirs(FLAGS.tmp_ims)

    if needCrop:
        detector = dlib.get_frontal_face_detector()
        if not os.path.exists(FLAGS.tmp_detect):
            # shutil.rmtree('tmp_detect')
            os.makedirs(FLAGS.tmp_detect)

    output_dir = os.path.abspath(FLAGS.savePath)
    image_path = os.path.abspath(FLAGS.imagePath)
    print("imagePath:{0}, outpath:{1}".format(image_path, output_dir))

    start_time = time.time()
    img = prepareImage(image_path)  # prepare Image
    if img is None:
        print("Did not detected face ")
        exit()
    cnn_3dmm(img, output_dir)

    print("Cost time {0} Sec.".format(
        time.time() - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--imagePath",
        dest = "imagePath",
        default = "./test_image.jpg",
        type = str,
        help ="The image need to be reconstruction!" )

    parser.add_argument(
        "-s",
        "--savePath",
        dest = "savePath",
        default = "./",
        type = str,
        help ="the floder to output the model" )

    parser.add_argument(
        "--tmp_ims",
        dest = "tmp_ims",
        default = "./tmp_ims",
        type = str,
        help ="the floder to put temp images!" )

    parser.add_argument(
        "--tmp_detect",
        dest = "tmp_detect",
        default = "./tmp_detect",
        type = str,
        help ="the floder to put temp detected!" )

    # CNN network spec
    parser.add_argument(
        "--deploy_path",
        dest = "deploy_path",
        default = "../CNN/deploy_network.prototxt",
        type = str,
        help ="Path to deploy_network.prototxt" )

    # CNN network spec
    parser.add_argument(
        "--model_path",
        dest = "model_path",
        default = "../CNN/3dmm_cnn_resnet_101.caffemodel",
        type = str,
        help ="Path to deploy_network.prototxt" )

    # CNN network spec
    parser.add_argument(
        "--mean_path",
        dest = "mean_path",
        default = "../CNN/mean.binaryproto",
        type = str,
        help ="Path to mean.binaryproto" )

    ## Modifed Basel Face Model
    parser.add_argument(
        "--BFM_path",
        dest = "BFM_path",
        default = "../3DMM_model/BaselFaceModel_mod.mat",
        type = str,
        help ="Path to BaselFaceModel_mod.mat" )

    parser.add_argument(
        "--predictor_path",
        dest = "predictor_path",
        default = "../dlib_model/shape_predictor_68_face_landmarks.dat",
        type = str,
        help ="Path to shape_predictor_68_face_landmarks.dat" )

    FLAGS, unparsed = parser.parse_known_args()

    print( FLAGS )      # print the arguments
    main()
