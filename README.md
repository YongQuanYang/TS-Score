This is the source codes for the generation of TS-Score proposed in our paper.

Dependencies:
python 3.5
tensorflow 1.10.0

CNN-I provides the pre-trained deep CNN segmentation model which was used to generate the stroma regions for pCR scoring in our paper.
    The script Seg_4_VisualCmp.py is for visualized comparison of the predictions of the pre-trained deep CNN segmentation model with mannually labeled ground-truth.
    The script Seg_4_pCR is for generating the stroma regions for pCR scoring using the pre-trained deep CNN segmentation model.

CNN-II provides the pre-trained CNN model which was used to generate the TS-Score in our paper.
    
