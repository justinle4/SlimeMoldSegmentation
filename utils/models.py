import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.lin = nn.Linear(3, 2)

    def forward(self, x):
        s = self.lin(x.transpose(1, 3)).transpose(1, 3)
        return s


class OneConv_kernel3(nn.Module):
    def __init__(self):
        super(OneConv_kernel3, self).__init__()
        self.conv1 = nn.Conv2d(3, 2, 3, padding=1)

    def forward(self, x):
        s = self.conv1(x)
        return s


class TwoConv_kernel3(nn.Module):
    def __init__(self):
        super(TwoConv_kernel3, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3, padding=1)
        self.conv2 = nn.Conv2d(10, 2, 3, padding=1)

    def forward(self, x):
        z = F.relu(self.conv1(x))
        s = self.conv2(z)
        return s


class TwoConv_kernel3_moreChannels(nn.Module):
    def __init__(self):
        super(TwoConv_kernel3_moreChannels, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 3, padding=1)
        self.conv2 = nn.Conv2d(20, 2, 3, padding=1)

    def forward(self, x):
        z = F.relu(self.conv1(x))
        s = self.conv2(z)
        return s


class TwoConv_kernel3_manyChannels(nn.Module):
    def __init__(self):
        super(TwoConv_kernel3_manyChannels, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, 3, padding=1)
        self.conv2 = nn.Conv2d(50, 2, 3, padding=1)

    def forward(self, x):
        z = F.relu(self.conv1(x))
        s = self.conv2(z)
        return s


class ThreeConv_kernel3(nn.Module):
    def __init__(self):
        super(ThreeConv_kernel3, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3, padding=1)
        self.conv2 = nn.Conv2d(10, 5, 3, padding=1)
        self.conv3 = nn.Conv2d(5, 2, 3, padding=1)

    def forward(self, x):
        z = F.relu(self.conv1(x))
        w = F.relu(self.conv2(z))
        s = self.conv3(w)
        return s


class ThreeConv_kernel3_moreChannels(nn.Module):
    def __init__(self):
        super(ThreeConv_kernel3_moreChannels, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 3, padding=1)
        self.conv2 = nn.Conv2d(20, 10, 3, padding=1)
        self.conv3 = nn.Conv2d(10, 2, 3, padding=1)

    def forward(self, x):
        z = F.relu(self.conv1(x))
        w = F.relu(self.conv2(z))
        s = self.conv3(w)
        return s


class FourConv_kernel3(nn.Module):
    def __init__(self):
        super(FourConv_kernel3, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 3, padding=1)
        self.conv2 = nn.Conv2d(20, 10, 3, padding=1)
        self.conv3 = nn.Conv2d(10, 5, 3, padding=1)
        self.conv4 = nn.Conv2d(5, 2, 3, padding=1)

    def forward(self, x):
        z = F.relu(self.conv1(x))
        w = F.relu(self.conv2(z))
        v = F.relu(self.conv3(w))
        s = self.conv4(v)
        return s


class OneConv_kernel5(nn.Module):
    def __init__(self):
        super(OneConv_kernel5, self).__init__()
        self.conv1 = nn.Conv2d(3, 2, 5, padding=2)

    def forward(self, x):
        s = self.conv1(x)
        return s


class TwoConv_kernel5(nn.Module):
    def __init__(self):
        super(TwoConv_kernel5, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5, padding=2)
        self.conv2 = nn.Conv2d(10, 2, 5, padding=2)

    def forward(self, x):
        z = F.relu(self.conv1(x))
        s = self.conv2(z)
        return s


class TwoConv_kernel5_moreChannels(nn.Module):
    def __init__(self):
        super(TwoConv_kernel5_moreChannels, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, padding=2)
        self.conv2 = nn.Conv2d(20, 2, 5, padding=2)

    def forward(self, x):
        z = F.relu(self.conv1(x))
        s = self.conv2(z)
        return s


class TwoConv_kernel5_manyChannels(nn.Module):
    def __init__(self):
        super(TwoConv_kernel5_manyChannels, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, 5, padding=2)
        self.conv2 = nn.Conv2d(50, 2, 5, padding=2)

    def forward(self, x):
        z = F.relu(self.conv1(x))
        s = self.conv2(z)
        return s


class ThreeConv_kernel5(nn.Module):
    def __init__(self):
        super(ThreeConv_kernel5, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5, padding=2)
        self.conv2 = nn.Conv2d(10, 5, 5, padding=2)
        self.conv3 = nn.Conv2d(5, 2, 5, padding=2)

    def forward(self, x):
        z = F.relu(self.conv1(x))
        w = F.relu(self.conv2(z))
        s = self.conv3(w)
        return s


class ThreeConv_kernel5_moreChannels(nn.Module):
    def __init__(self):
        super(ThreeConv_kernel5_moreChannels, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, padding=2)
        self.conv2 = nn.Conv2d(20, 10, 5, padding=2)
        self.conv3 = nn.Conv2d(10, 2, 5, padding=2)

    def forward(self, x):
        z = F.relu(self.conv1(x))
        w = F.relu(self.conv2(z))
        s = self.conv3(w)
        return s


class FourConv_kernel5(nn.Module):
    def __init__(self):
        super(FourConv_kernel5, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, padding=2)
        self.conv2 = nn.Conv2d(20, 10, 5, padding=2)
        self.conv3 = nn.Conv2d(10, 5, 5, padding=2)
        self.conv4 = nn.Conv2d(5, 2, 5, padding=2)

    def forward(self, x):
        z = F.relu(self.conv1(x))
        w = F.relu(self.conv2(z))
        v = F.relu(self.conv3(w))
        s = self.conv4(v)
        return s


class ThreeConv_kernel3_resNet(nn.Module):
    def __init__(self):
        super(ThreeConv_kernel3_resNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 5, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(5)
        self.conv3 = nn.Conv2d(5, 2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(2)

    def forward(self, x):
        z1 = self.bn1(F.relu(self.conv1(x)))
        z2 = self.bn2(F.relu(self.conv2(z1)))
        s = self.bn3(F.relu(self.conv3(z2)))
        return s


class UNet_simple(nn.Module):
    def __init__(self):
        super(UNet_simple, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(20, 40, 3)
        self.conv_transpose1 = nn.ConvTranspose2d(40, 20, 2, stride=2)
        self.conv4 = nn.Conv2d(20, 20, 3)
        self.conv_transpose2 = nn.ConvTranspose2d(20, 10, 2, stride=2)
        self.convFinal = nn.Conv2d(10, 2, 1)

    def forward(self, x):
        z1 = F.relu(self.conv1(x))
        z2 = self.pool1(z1)
        z3 = F.relu(self.conv2(z2))
        z4 = self.pool2(z3)
        z5 = F.relu(self.conv3(z4))
        z6 = F.relu(self.conv_transpose1(z5))
        z7 = F.relu(self.conv4(z6))
        z8 = F.relu(self.conv_transpose2(z7))
        z9 = self.convFinal(z8)
        s = F.interpolate(z9, x.size()[2:], mode='bilinear', align_corners=True)
        return s


class UNetEnc(nn.Module):

    def __init__(self, in_channels, features, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_channels, features, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features, out_channels, 2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class UNetDec(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers += [nn.Dropout(.5)]
        layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)


class UNet_simple_base(nn.Module):

    def __init__(self, num_channels_input=3, num_classes=2):
        super().__init__()

        self.dec1 = UNetDec(num_channels_input, 64)  # num_channels_input=3 -> RBG, ...=1 -> gray
        self.dec2 = UNetDec(64, 128)
        self.center = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.enc2 = UNetEnc(256, 128, 64)
        self.enc1 = nn.Sequential(
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        dec1 = self.dec1(x)
        dec2 = self.dec2(dec1)
        center = self.center(dec2)
        enc2 = self.enc2(torch.cat([
            center, F.interpolate(dec2, center.size()[2:], mode='bilinear', align_corners=True)], 1))
        enc1 = self.enc1(torch.cat([
            enc2, F.interpolate(dec1, enc2.size()[2:], mode='bilinear', align_corners=True)], 1))

        return F.interpolate(self.final(enc1), x.size()[2:], mode='bilinear', align_corners=True)


class UNet_full_base(nn.Module):

    def __init__(self, num_channels_input=3, num_classes=2):
        super().__init__()

        self.dec1 = UNetDec(num_channels_input, 64) # num_channels_input=3 -> RBG, ...=1 -> gray
        self.dec2 = UNetDec(64, 128)
        self.dec3 = UNetDec(128, 256)
        self.dec4 = UNetDec(256, 512, dropout=True)
        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.enc4 = UNetEnc(1024, 512, 256)
        self.enc3 = UNetEnc(512, 256, 128)
        self.enc2 = UNetEnc(256, 128, 64)
        self.enc1 = nn.Sequential(
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        dec1 = self.dec1(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)
        center = self.center(dec4)
        enc4 = self.enc4(torch.cat([
            #center, F.interpolate_bilinear(dec4, center.size()[2:])], 1))  # depracated
            center, F.interpolate(dec4, center.size()[2:], mode='bilinear', align_corners=True)], 1))
        enc3 = self.enc3(torch.cat([
            enc4, F.interpolate(dec3, enc4.size()[2:], mode='bilinear', align_corners=True)], 1))
        enc2 = self.enc2(torch.cat([
            enc3, F.interpolate(dec2, enc3.size()[2:], mode='bilinear', align_corners=True)], 1))
        enc1 = self.enc1(torch.cat([
            enc2, F.interpolate(dec1, enc2.size()[2:], mode='bilinear', align_corners=True)], 1))

        return F.interpolate(self.final(enc1), x.size()[2:], mode='bilinear', align_corners=True)


class FCN_Resnet50(nn.Module):
    def __init__(self):
        super(FCN_Resnet50, self).__init__()
        self.model = models.segmentation.fcn_resnet50(weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT)
        for param in self.model.parameters(): # freeze all layers
            param.requires_grad = False
        
        self.model.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
        self.model.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        s = self.model(x)
        return s['out']


class FCN_Resnet101(nn.Module):
    def __init__(self):
        super(FCN_Resnet101, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet101', pretrained=True)
        for param in self.model.parameters():  # freeze all layers
            param.requires_grad = False

        self.model.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
        self.model.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        s = self.model(x)
        return s['out']
