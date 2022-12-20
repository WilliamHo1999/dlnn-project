# dlnn-project

## 1. APPROACH

Attacks:
- PGD: Projected Gradient Descent PGD
- FGSM: Fast gradient sign method
- Universal Pertubation

Interpretation methods
- GradCam

Datasets
- MNIST (bw digit image classification, 10 classes)
- Sub set of ImageNet (3-channel general inage classification, 100 classes, 135,000 samples)
- Pascal VOC

Models 
- LeNet inspired CNN (for MNIST)
- ResNet (for ImageNet)


## 2. STRUCTURE
0. /utils/ :
This folder contains anything useful that makes our lives easier, and is not directly related related to the other folders.

1. /data/ :
holds data sets.
This folder contains datasets, and the neccesary classes to load data to a PyTorch pipeline.

The actual data should be in the root folder under another folder ```Data/```

2. training :
-- /training/ This folder contains the training script in the form of a class. The trainer takes a model, optimizer, scheduler etc., and trains the network for a given amount of epochs.
-- /fine_tune_resnet_100_class_images.ipynb fine-tunes the original ResNet work in which the last layer has been replaced to correspond to a 100-class classification

3. adversarial attacks
-- /attacks/: python files with classes for different attacks
-- /notebook_attacks.ipynb: proof of concept of FGS attack

4. explainability
-- /explainability/ : This folder contains classes for producing explanations of neural networks, mostly in the form of saliency maps.:

5. adversarial training
-- /FineTuneAdvTraining.py is the script to be run on a VM to run the adversarial training (PGD-AT) on the network




