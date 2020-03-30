# Face mask detection


This model is a lightweight face mask detection model. Based on ssd,the backbone is Mobilenet and RFB.
 

## Key Features

- [x] Tensorflow 2.1
- [x] Trainging and Inference
- [x] Precision with mAP
- [x] Eager mode training with `tf.GradientTape`
- [x] Network function with `tf.keras`
- [x] Dataset prepocessing with `tf.data.TFRecordDataset`

```bash
├── assert
│   ├── 1_Handshaking_Handshaking_1_71.jpg
│   ├── out_1_Handshaking_Handshaking_1_71.jpg
│   ├── out_test_00002330.jpg
│   └── test_00002330.jpg
├── checkpoints
│   └── weights_epoch_120.h5
├── components
│   ├── config.py
│   ├── __init__.py
│   ├── kmeans.py
│   ├── prior_box.py
│   └── utils.py
├── dataset
│   ├── check_dataset.py
│   ├── tf_dataset_preprocess.py
│   ├── train_mask.tfrecord
│   ├── trainval_mask.tfrecord
│   ├── val_mask.tfrecord
│   ├── voc_to_tfrecord.py
├── inference.py
├── logs
│   └── train
├── mAP
│   ├── compute_mAP.py
│   ├── detection-results
│   ├── detect.py
│   ├── ground-truth
│   ├── __init__.py
│   ├── map-results
│   └── README.md
├── Maskdata
│   ├── Annotations
│ 	 ├── ImageSets
│   	   └── Main
│   │       ├── train.txt
│   │       ├── trainval.txt
│   │       └── val.txt
│   └── JPEGImages
├── network
│   ├── __init__.py
│   ├── losses.py
│   ├── model.py
│   ├── net.py
│   ├── network.py
├── README.md
└── train.py
```

## Usage

### Installation

Create a new python virtual environment by [Anaconda](https://www.anaconda.com/) ,`pip install -r requirements.txt`

### Data Preparing

1. Face Mask Data

   Source data from  [**AIZOOTech**](https://github.com/AIZOOTech/FaceMaskDetection)  , which is a great job. 

   I checked  and corrected some error to apply my own training network according to the voc dataset format. You can download it here. [https://pan.baidu.com/s/1330OWyOL4huB-GWXRkKK6Q](https://pan.baidu.com/s/1330OWyOL4huB-GWXRkKK6Q) code：x5e0

2. Data Processing

   + Download the mask data images 

   + Convert the training images and annotations to tfrecord file with the the script bellow.

     ```bash
     python dataset/voc_to_tfrecord.py --dataset_path Maskdata/  --output_file dataset/train_mask.tfrecord --split train
     ```

     you can change the --split parameters to 'val' to get the validation tfrecord, Please modify the inside setting `voc_to_tfrecord.py` for different situations.

3. Check tfrecord dataloader by run `python dataset/check_dataset.py` .

### Training

1. Modify your configuration in `components/config.py`. 

   You can get the anchors by run `python components/kmeans.py`

2. Train the model by run `python train.py` .

### Inference

+ Run on video

  ```bash
  python inference.py  --model_path checkpoints/ --camera True
  or
  python inference.py  --model_path checkpoints/*.h5 --camera True
  ```

+ Detect on Image

  ```bash
  python inference.py  --model_path checkpoints/*.h5 --img_path assert/1_Handshaking_Handshaking_1_71.jpg
  ```

  ![](https://raw.githubusercontent.com/PureHing/face-mask-detection-tf2/master/assert/out_test_00002330.jpg)

  ![](https://raw.githubusercontent.com/PureHing/face-mask-detection-tf2/master/assert/out_1_Handshaking_Handshaking_1_71.jpg)
  

### mAP

+ Convert xml to txt file on  `mAP/ground truth`, predicting  the bbox and class on `mAP/detection-results`.

  ```bash
   python mAP/detect.py --model_path checkpoints/*.h5 --dataset_path Maskdata/ --split val 
  
   python mAP/compute_mAP.py
  ```

