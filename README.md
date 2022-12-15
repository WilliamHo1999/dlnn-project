# dlnn-project


## STRUCTURE
0. /utils/ :
This folder contains anything useful that makes our lives easier, and is not directly related related to the other folders.

1. /data/ :
holds data sets.
This folder contains datasets, and the neccesary classes to load data to a PyTorch pipeline.

The actual data should be in the root folder under another folder ```Data/```

2. /training/ :
This folder contains the training script in the form of a class. The trainer takes a model, optimizer, scheduler etc., and trains the network for a given amount of epochs.

3. attacks
-- /attacks/: python files with classes for different attacks
-- /notebook_attacks.ipynb: proof of concept of FGS attack

4. explainability
-- /explainability/ : This folder contains classes for producing explanations of neural networks, mostly in the form of saliency maps.:




