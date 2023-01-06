from utils.models import *
from utils.training_module import *
import matplotlib.pyplot as plt

# declaring hyper parameters
cfg = {
    'num_epochs': 1,  # 40 epochs for real training, 5 for debugging
    'learning_rate': 0.01,  # 10^-4 for UNET, 10^-2 for CNN
    'nbr_classes': 2,
    'weight': [1.0, 1.0],
    'batch_size': 4,  # batch size of 32 for real training, 4 for debugging
    'pct_train_set': .8,
    'shuffle_dataset': True,
    'show_mask': True,  # True for real training, False for debugging
    'name_classes': ['background', 'slime'],
    'num_workers': 0,  # num_workers of 4 for training on Agave, 0 on a Windows machine
    # agave result folder: '../results_ML'
    # personal machine result folder: "C:\\Users\\lejus\\PycharmProjects\\Fall2022Project\\Results
    'path_data': "C:\\Users\\lejus\\PycharmProjects\\Fall2022Project\\Cropped Training Data\\",
    # agave path data: "../Slime_Mold_Project/Slime_Mold_Dataset/"
    # personal machine path data: "C:\\Users\\lejus\\PycharmProjects\\Fall2022Project\\Cropped Training Data\\"
    'model': 'FCN_Resnet101'  # see utils\models.py for full list of models
}
cfg['folder_result'] = 'C:\\Users\\lejus\\PycharmProjects\\Fall2022Project\\Results\\' + cfg['model']

cfg['transform_train'] = ("Compose(["
                          "RandomHorizontalFlip(),"
                          "RandomVerticalFlip(),"
                          "Resize(224),"
                          "CenterCrop(224),"
                          "ToTensor()])")
cfg['transform_test'] = ("Compose(["
                         "Resize(224),"
                         "CenterCrop(224),"
                         "ToTensor()])")

# training
myModel = eval(cfg['model'])()
df_training = trainModel(myModel, cfg)
