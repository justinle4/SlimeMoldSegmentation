Performs semantic segmentation on a slime mold image dataset. Adjust hyperparameters such as path data, batch size, model, and learning rate through main_network.py
Perform segmentation inference through box_slime_inference.py

Miniature datasets are provided for training two neural networks: 
- boxNet: creates a bounding box around the petri dish containing slime mold.
- slimeNet: segments the slime mold inside an image of a petri dish.

Datasets can be found in the "datasets" folder.
All results from training and inference will be saved in the results_ML folder.

Example outputs from segmentation inference:

![alt text](https://github.com/justinle4/SlimeMoldSegmentation/blob/master/sample_output/Protocole10__B2__ConJ5ExB4_prediction.png?raw=true) \
\
![alt text](https://github.com/justinle4/SlimeMoldSegmentation/blob/master/sample_output/Protocole15__MALU__ExpJ10ExB4_prediction.png?raw=true)
