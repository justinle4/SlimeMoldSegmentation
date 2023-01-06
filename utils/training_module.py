import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import *
import torch.nn.functional as F
# classic
import sys
from PIL import Image
import numpy as np
import pandas as pd
import time, os, datetime
# personal
from utils.slime_loader_update_crop_data import *
from utils.toolbox import *


# returns a list of the jaccard indices of a batch
def jaccard_index_batch(pred_mask, mask):
    num = np.minimum(pred_mask, mask)
    num_sum = np.sum(np.sum(num, axis=2), axis=1)
    den = np.maximum(pred_mask, mask)
    den_sum = np.sum(np.sum(den, axis=2), axis=1)
    return num_sum / den_sum


def trainModel(f, cfg):
    myDataSet = SlimeDataSet(cfg['path_data'], eval(cfg['transform_train']), eval(cfg['transform_test']),
                             cfg['pct_train_set'], cfg['shuffle_dataset'])
    if myDataSet.__len__() == 0:
        print("--- Problem initialization: dataSet empty\n")
        sys.exit(2)

    dataLoader_train = DataLoader(myDataSet, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                                  sampler=myDataSet.train_sampler)
    dataLoader_valid = DataLoader(myDataSet, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'],
                                  sampler=myDataSet.test_sampler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    myModel = torch.nn.DataParallel(f)
    myModel.to(device)

    weight = torch.tensor(cfg['weight'])
    criterion = nn.CrossEntropyLoss(weight=weight.to(device))
    optimizer = torch.optim.Adam(myModel.parameters(), lr=cfg['learning_rate'])

    # training loop
    t0 = time.time()
    df = pd.DataFrame(columns=('epoch', 'loss_train', 'loss_test', 'jaccard_train', 'jaccard_test'))
    print("--- Training Begins ---")
    for epoch in range(cfg['num_epochs']):

        if epoch == 2:
            for param in myModel.parameters():
                param.requires_grad = True

        myModel.train()
        list_loss_train, list_jaccard_train = [], []
        for X, mask in dataLoader_train:
            # forward pass
            yhat = myModel(X.to(device))

            # loss
            loss = criterion(yhat, mask.to(device))
            list_loss_train.append(loss.item())

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accuracy
            predicted_mask = yhat.cpu().detach().numpy()[:, 1, :, :] > yhat.cpu().detach().numpy()[:, 0, :, :]
            list_jaccard_train.extend(jaccard_index_batch(predicted_mask.astype(int), mask.cpu().numpy()))

        # evaluate model
        myModel.eval()
        list_loss_test, list_jaccard_test = [], []
        with torch.no_grad():
            for X, mask in dataLoader_valid:
                yhat = myModel(X.to(device))
                list_loss_test.append(criterion(yhat, mask.to(device)).item())
                predicted_mask = yhat.cpu().detach().numpy()[:, 1, :, :] > yhat.cpu().detach().numpy()[:, 0, :, :]
                list_jaccard_test.extend(jaccard_index_batch(predicted_mask.astype(int), mask.cpu().numpy()))

        # insert results into dataframe
        df.loc[epoch] = [epoch, np.mean(list_loss_train), np.mean(list_loss_test),
                         np.mean(list_jaccard_train), np.mean(list_jaccard_test)]
        print(f"Epoch {epoch + 1} complete")

    seconds_elapsed = time.time() - t0
    cfg['time_training'] = '{:.0f} minutes, {:.0f} seconds'.format(seconds_elapsed // 60, seconds_elapsed % 60)
    print(f"--- Training Complete in {cfg['time_training']}  ---")
    cfg['str_time'] = datetime.datetime.now().replace(microsecond=0).isoformat(sep='_').replace(':', 'h', 1).replace(
        ':', 'm', 1)

    save_df_network(myModel, cfg, df)
    if cfg['show_mask']:
        os.makedirs(cfg['folder_result'] + '/Report_' + cfg['str_time'] + "/masks")
        os.makedirs(cfg['folder_result'] + '/Report_' + cfg['str_time'] + "/plots")
        dataSet_prediction = SlimeDataSet_prediction(myDataSet.filenames, eval(cfg['transform_test']),
                                                     myDataSet.train_sampler)
        with torch.no_grad():
            N = myDataSet.__len__()
            for k in range(N):
                X, y, filename, inTestSet = dataSet_prediction.__getitem__(k)
                score = myModel(X.unsqueeze(0).to(device))
                proba = F.softmax(score, dim=1)
                plot_save_mask(proba[0, 1, :, :].cpu().numpy(),
                               cfg['folder_result'] + '/Report_' + cfg[
                                   'str_time'] + '/masks/' + 'mask_' + str(k) + '_inTest' + str(inTestSet) + '.jpg')
                plot_save_proba(proba[0, 1, :, :].cpu().numpy(),
                                cfg['folder_result'] + '/Report_' + cfg[
                                    'str_time'] + '/plots/' + 'plot_' + str(k) + '_inTest' + str(inTestSet) + '.jpg')

    return df
