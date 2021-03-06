# 3DImHistNet: Learnable 3D Image Histogram-based DNN

This code is the 3D extention of 2D ImHistNet proposed in this [paper](http://www.cs.sfu.ca/~hamarneh/ecopy/miccai2019e.pdf) in MICCAI 2019. This 3D module can be incorporated into a conventional CNN. For details about the algorithm, you can check this [repository](https://github.com/marafathussain/ImHistNet).

## Usage
The 3D ImHistNet module is written in ```imhistnet.py``` file. It can be plugged in to any conventional 3D CNN. An example code ```train_3d_imhistnet.ipynb``` is given in a [MONAI](https://monai.io/) framework. Please note that this notebook code is only clarify the usage. Also please make sure that you have add the following lines in your main training code after calling the model, as this network fixes the kernel parameter for the first Convolution layer and bias of the second Convolution layer to value 1 and does not update during backpropagation:

```
model = imhistnet_3d().to(device)
for name, param in model.named_parameters():
    if name in ['conv1.weight', 'conv2.bias']:
        param.required_grad = False
```

## Citations
If you find this work useful, please cite the first or all of the following original papers:

```
@article{hussain2021learnable,
  title={Learnable Image Histograms-based Deep Radiomics for Renal Cell Carcinoma Grading and Staging},
  author={Hussain, Mohammad Arafat and Hamarneh, Ghassan and Garbi, Rafeef},
  journal={Computerized Medical Imaging and Graphics},
  pages={101924},
  year={2021},
  publisher={Elsevier}
}
```

```
@inproceedings{hussain2019imhistnet,
  title={ImHistNet: Learnable Image Histogram Based DNN with Application to Noninvasive Determination of Carcinoma Grades in CT Scans},
  author={Hussain, Mohammad Arafat and Hamarneh, Ghassan and Garbi, Rafeef},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={1-9},
  year={2019},
  organization={Springer}
}
```

```
@inproceedings{hussain2019renal,
  title={Renal Cell Carcinoma Staging with Learnable Image Histogram-based Deep Neural Network},
  author={Hussain, Mohammad Arafat and Hamarneh, Ghassan and Garbi, Rafeef},
  booktitle={International Workshop on Machine Learning in Medical Imaging (MLMI)},
  pages={1-8},
  year={2019},
  organization={Springer}
}
```
