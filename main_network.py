from utils.models import *
from utils.training_module import *
import matplotlib.pyplot as plt

# declaring hyper parameters
cfg = {
    'boxOrslime': 1,  # 1: Box, 2: Slime
    'num_epochs': 40,
    'learning_rate': 0.0001,  # Use 10^-4 for larger models like UNET and FCN Resnet and 10^-2 for smaller models like CNN.
    'nbr_classes': 2,
    'weight': [0.5, 0.5],
    'batch_size': 8,  # For larger models, use a smaller batch size such as 8 or 16. For smaller models, a batch size of up to 32 can be used.
    'pct_train_set': .8,
    'shuffle_dataset': True,
    'show_mask': True,
    'name_classes': ['background', 'slime'],
    'num_workers': 4,
    'path_data': "INSERT PATH DATA HERE",  # Insert path where dataset is located. Dataset must contain
                                           # slime mold images and a mask for each image.
    'model': 'FCN_Resnet50'  # See utils/models.py for a complete list of trainable models.
}
cfg['folder_result'] = 'results_ML/' + cfg['model']

if cfg['boxOrslime'] == 1:  # Use BoxNet
    cfg['transform_train'] = ("Compose(["
                              "RandomHorizontalFlip(),"
                              "RandomVerticalFlip(),"
                              "Resize(599, max_size=600),"
                              "CenterCrop(600),"
                              "ToTensor()])")
    cfg['transform_test'] = ("Compose(["
                             "Resize(599, max_size=600),"
                             "CenterCrop(600),"
                             "ToTensor()])")

else:  # Use Slimenet
    cfg['transform_train'] = ("Compose(["
                              "RandomHorizontalFlip(),"
                              "RandomVerticalFlip(),"
                              "Resize(400),"
                              "CenterCrop(400),"
                              "ToTensor()])")
    cfg['transform_test'] = ("Compose(["
                             "Resize(400),"
                             "CenterCrop(400),"
                             "ToTensor()])")

# training
myModel = eval(cfg['model'])()
df_training = trainModel(myModel, cfg)
print("--- Main network has fully completed running ---")

