Performs semantic segmentation on a slime mold image dataset. Adjust hyperparameters such as path data, batch size, model, and learning rate through main_network.py
Perform segmentation inference through box_slime_inference.py

Miniature datasets are provided for training two neural networks: 
- boxNet: creates a bounding box around the petri dish containing slime mold.
- slimeNet: segments the slime mold inside an image of a petri dish.

Datasets can be found in the "datasets" folder.
All results from training and inference will be saved in the results_ML folder.
