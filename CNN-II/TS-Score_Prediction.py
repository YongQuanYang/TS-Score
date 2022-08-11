# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
from tensorflow.keras.models import load_model
import os
import sys
from shutil import copyfile


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

sys.path.append("..")


def img_preprocess(img_path):
    img = Image.open(img_path)

    if np.asarray(img).shape[2] == 4:
        img = Image.open(img_path).convert("RGB")

    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img, mode='tf')
    return img


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    model = load_model('./ckpt/release.h5')
    
    test_inffer = './testing_inffer'
     
    dirs = os.listdir('./testing')
    wsi_label_imgs = []
    for p in dirs:
        wsi_label_imgs.append(os.path.join('./testing',p))
    
    for d in wsi_label_imgs:
        newwsidir = os.path.join(test_inffer, d.split('/')[-1])

        imgps = os.listdir(d)

        if not os.path.exists(newwsidir):
            os.makedirs(newwsidir)
        
        for ip in imgps:
            img = img_preprocess(os.path.join(d,ip))
            prob_list = model.predict(img)[0]
            prob_list = prob_list.tolist()
            index = prob_list.index(max(prob_list))
            mp = max(prob_list)
            dimgn = d.split('/')[-1]+'_'+str(prob_list[0])+'_'+str(prob_list[1])+'.png'
            destp = os.path.join(newwsidir, dimgn)
            copyfile(os.path.join(d,ip), destp)
