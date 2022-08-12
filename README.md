This project maintains the source codes for the generation of the TS-Score proposed in our paper.


1. CNN-I

Dependencies: python 3.5, tensorflow 1.10.0

The CNN-I folder provides the pre-trained deep CNN segmentation model (available at: https://pan.baidu.com/s/1SWH1PEvlIfttu-aotjoF7g?pwd=ckpt) and corresponding python scripts that were used to generate the stroma regions for pCR scoring in our paper.

The script Seg_4_VisualCmp.py is for visualized comparison of the predictions of the pre-trained deep CNN segmentation model with mannually labeled ground-truth.
    
The script Seg_4_pCR.py is for generating the stroma regions for pCR scoring using the pre-trained deep CNN segmentation model.

Our paper for the details of the construction of CNN-I is available at: https://arxiv.org/ftp/arxiv/papers/2110/2110.10325.pdf 


2. CNN-II

Dependencies: python 3.5, tensorflow 2.3.0

The CNN-II folder provides the pre-trained deep CNN pCR scoring model (available at: https://pan.baidu.com/s/1SWH1PEvlIfttu-aotjoF7g?pwd=ckpt) and corresponding python scripts that were used to generate the TS-Score proposed in our paper, based on the stroma regions proposed by CNN-I.

The script TS-Score_Prediction.py is for scoring each path of stroma regions proposed by CNN-I.
    
The script TS-Score_Generation.py is for generating the TS-Score, based on the scored pathces of stroma regions proposed by CNN-I.
    
