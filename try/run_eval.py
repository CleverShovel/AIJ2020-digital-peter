import torch.nn.functional as F
import torch.nn as nn
import torch

import torchvision.transforms.functional as VF

from PIL import Image

import numpy as np

import os
from os.path import join
from collections import Counter

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import math

from ctcdecode import CTCBeamDecoder

import multiprocessing

n_cpus = multiprocessing.cpu_count()

# letters = [' ', ')', '+', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']', 'i', 'k', 'l', '|', '×', 'ǂ',
#            'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х',
#            'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'і', 'ѣ', '–', '…', '⊕', '⊗']
letters = list(' ()+/0123456789[]abdefghiklmnoprstu|×ǂабвгдежзийклмнопрстуфхцчшщъыьэюяѣ–⊕⊗')

std, mean = (0.3847, 0.3815, 0.3763), (0.6519, 0.6352, 0.5940)

def process_image(img):
    img = VF.resize(img, 128)
    img = VF.pad(img, (0, 0, max(1024 - img.size[0], 0), max(128 - img.size[1], 0)))
    img = VF.resize(img, (128, 1024))
    img = VF.to_tensor(img)
    img = VF.normalize(img, mean, std)

    return img


# CNN-BLSTM
class CNNBLSTM(nn.Module):
    def __init__(self, conv_drop=0.2, lstm_drop=0.5):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(16)
        self.lelu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(32)
        self.lelu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        self.dropout3 = nn.Dropout2d(conv_drop)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(48)
        self.lelu3 = nn.LeakyReLU()
        self.pool3 = nn.MaxPool2d(2)

        self.dropout4 = nn.Dropout2d(conv_drop)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, padding=1)
        self.norm4 = nn.BatchNorm2d(64)
        self.lelu4 = nn.LeakyReLU()

        self.dropout5 = nn.Dropout2d(conv_drop)
        self.conv5 = nn.Conv2d(64, 80, kernel_size=3, padding=1)
        self.norm5 = nn.BatchNorm2d(80)
        self.lelu5 = nn.LeakyReLU()

        self.flatten1 = nn.Flatten(1, 2)

        self.dropout6 = nn.Dropout(lstm_drop)
        self.lstm6 = nn.LSTM(80*16, hidden_size=256, num_layers=3, dropout=lstm_drop, bidirectional=True, batch_first=True)

        self.dropout7 = nn.Dropout(lstm_drop)
        self.linear = nn.Linear(2*256, len(letters) + 1)
    
    def forward(self, x):
        x = self.pool1(self.lelu1(self.norm1(self.conv1(x))))
        x = self.pool2(self.lelu2(self.norm2(self.conv2(x))))
        x = self.pool3(self.lelu3(self.norm3(self.conv3(self.dropout3(x)))))
        x = self.lelu4(self.norm4(self.conv4(self.dropout4(x))))
        x = self.lelu5(self.norm5(self.conv5(self.dropout5(x))))

        x = self.flatten1(x)
        x = x.transpose(1, 2)

        x, _ = self.lstm6(x)
        x = self.dropout7(x)
        x = self.linear(x)

        return x

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == FullGatedConv2d:
        nn.init.kaiming_uniform_(m.weight)


# # https://github.com/mf1024/Batch-Renormalization-PyTorch/blob/master/batch_renormalization.py
# # Batch Renormalization for convolutional neural nets (2D) implementation based
# # on https://arxiv.org/abs/1702.03275

# class BatchNormalization2D(nn.Module):

    # def __init__(self, num_features,  eps=1e-05, momentum = 0.1):

        # super().__init__()

        # self.eps = eps
        # self.momentum = torch.tensor( (momentum), requires_grad = False)

        # self.gamma = nn.Parameter(torch.ones((1, num_features, 1, 1), requires_grad=True))
        # self.beta = nn.Parameter(torch.zeros((1, num_features, 1, 1), requires_grad=True))

        # self.running_avg_mean = torch.ones((1, num_features, 1, 1), requires_grad=False)
        # self.running_avg_std = torch.zeros((1, num_features, 1, 1), requires_grad=False)

    # def forward(self, x):

        # device = self.gamma.device

        # batch_ch_mean = torch.mean(x, dim=(0,2,3), keepdim=True).to(device)
        # batch_ch_std = torch.clamp(torch.std(x, dim=(0,2,3), keepdim=True), self.eps, 1e10).to(device)

        # self.running_avg_std = self.running_avg_std.to(device)
        # self.running_avg_mean = self.running_avg_mean.to(device)
        # self.momentum = self.momentum.to(device)

        # if self.training:

            # x = (x - batch_ch_mean) / batch_ch_std
            # x = x * self.gamma + self.beta

        # else:

            # x = (x - self.running_avg_mean) / self.running_avg_std
            # x = self.gamma * x + self.beta

        # self.running_avg_mean = self.running_avg_mean + self.momentum * (batch_ch_mean.data.to(device) - self.running_avg_mean)
        # self.running_avg_std = self.running_avg_std + self.momentum * (batch_ch_std.data.to(device) - self.running_avg_std)

        # return x


# class BatchRenormalization2D(nn.Module):

    # def __init__(self, num_features,  eps=1e-05, momentum=0.01, r_d_max_inc_step = 0.0001):
        # super().__init__()

        # self.eps = eps
        # self.momentum = torch.tensor( (momentum), requires_grad = False)

        # self.gamma = nn.Parameter(torch.ones((1, num_features, 1, 1)), requires_grad=True)
        # self.beta = nn.Parameter(torch.zeros((1, num_features, 1, 1)), requires_grad=True)

        # self.running_avg_mean = torch.ones((1, num_features, 1, 1), requires_grad=False)
        # self.running_avg_std = torch.zeros((1, num_features, 1, 1), requires_grad=False) 

        # self.max_r_max = 3.0
        # self.max_d_max = 5.0

        # self.r_max_inc_step = r_d_max_inc_step
        # self.d_max_inc_step = r_d_max_inc_step

        # self.r_max = torch.tensor( (1.0), requires_grad = False)
        # self.d_max = torch.tensor( (0.0), requires_grad = False)

    # def forward(self, x):

        # device = self.gamma.device

        # batch_ch_mean = torch.mean(x, dim=(0,2,3), keepdim=True).to(device)
        # batch_ch_std = torch.clamp(torch.std(x, dim=(0,2,3), keepdim=True), self.eps, 1e10).to(device)

        # self.running_avg_std = self.running_avg_std.to(device)
        # self.running_avg_mean = self.running_avg_mean.to(device)
        # self.momentum = self.momentum.to(device)

        # self.r_max = self.r_max.to(device)
        # self.d_max = self.d_max.to(device)


        # if self.training:

            # r = torch.clamp(batch_ch_std / self.running_avg_std, 1.0 / self.r_max, self.r_max).to(device).data.to(device)
            # d = torch.clamp((batch_ch_mean - self.running_avg_mean) / self.running_avg_std, -self.d_max, self.d_max).to(device).data.to(device)

            # x = ((x - batch_ch_mean) * r )/ batch_ch_std + d
            # x = self.gamma * x + self.beta

            # if self.r_max < self.max_r_max:
                # self.r_max += self.r_max_inc_step * x.shape[0]

            # if self.d_max < self.max_d_max:
                # self.d_max += self.d_max_inc_step * x.shape[0]

        # else:

            # x = (x - self.running_avg_mean) / self.running_avg_std
            # x = self.gamma * x + self.beta

        # self.running_avg_mean = self.running_avg_mean + self.momentum * (batch_ch_mean.data.to(device) - self.running_avg_mean)
        # self.running_avg_std = self.running_avg_std + self.momentum * (batch_ch_std.data.to(device) - self.running_avg_std)

        # return x


# class FullGatedConv2d(nn.Conv2d):
#     def __init__(self, in_channels, **kwargs):
#         super().__init__(in_channels, in_channels * 2, **kwargs)

#         self.channels = in_channels
#         self.sigm = nn.Sigmoid()
    
#     def forward(self, x):
#         x = super().forward(x)
#         gated_x = self.sigm(x[:, self.channels:, :, :])
#         return x[:, :self.channels, :, :] * gated_x

# class HTRFlorConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size):
#         super().__init__()

#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
#         self.prelu = nn.PReLU()
#         self.br = nn.BatchNorm2d(out_channels)

#     def forward(self, x):
#         x = self.br(self.prelu(self.conv(x)))
#         return x

# class HTRFlor(nn.Module):
#     def __init__(self, htr_dropout=0.2, gru_dropout=0.5):
#         super().__init__()

#         self.conv_block1 = HTRFlorConvBlock(3, 16, kernel_size=(3, 3))
#         self.gconv1 = FullGatedConv2d(16, kernel_size=(3, 3), padding=1)
#         self.pool1 = nn.MaxPool2d((2, 2))

#         self.conv_block2 = HTRFlorConvBlock(16, 32, kernel_size=(3, 3))
#         self.gconv2 = FullGatedConv2d(32, kernel_size=(3, 3), padding=1)
#         self.pool2 = nn.MaxPool2d((2, 2))

#         self.conv_block3 = HTRFlorConvBlock(32, 40, kernel_size=(2, 4))
#         self.gconv3 = FullGatedConv2d(40, kernel_size=(3, 3), padding=1)
#         self.drop3 = nn.Dropout2d(htr_dropout)

#         self.conv_block4 = HTRFlorConvBlock(40, 48, kernel_size=(3, 3))
#         self.gconv4 = FullGatedConv2d(48, kernel_size=(3, 3), padding=1)
#         self.drop4 = nn.Dropout2d(htr_dropout)

#         self.conv_block5 = HTRFlorConvBlock(48, 56, kernel_size=(2, 4))
#         self.gconv5 = FullGatedConv2d(56, kernel_size=(3, 3), padding=1)
#         self.drop5 = nn.Dropout2d(htr_dropout)

#         self.conv_block6 = HTRFlorConvBlock(56, 64, kernel_size=(3, 3))
#         # self.pool = nn.MaxPool2d((1, 2))

#         self.flatten = nn.Flatten(1, 2)

#         self.drop7 = nn.Dropout(gru_dropout)
#         self.lstm7 = nn.LSTM(64*24, 128, num_layers=3, dropout=gru_dropout, bidirectional=True, batch_first=True)
#         # self.lstm7 = nn.LSTM(64*24, 128, bidirectional=True, batch_first=True)

#         # self.linear7 = nn.Linear(2*128, 256)

#         # self.drop8 = nn.Dropout(gru_dropout)
#         # self.lstm8 = nn.LSTM(256, 128, bidirectional=True, batch_first=True)

#         self.linear8 = nn.Linear(2*128, len(letters) + 1)

#     def forward(self, x):
#         x = self.conv_block1(x)
#         x = self.pool1(self.gconv1(x))

#         x = self.conv_block2(x)
#         x = self.pool2(self.gconv2(x))

#         x = self.conv_block3(x)
#         x = self.drop3(self.gconv3(x))

#         x = self.conv_block4(x)
#         x = self.drop4(self.gconv4(x))

#         x = self.conv_block5(x)
#         x = self.drop5(self.gconv5(x))

#         x = self.flatten(self.conv_block6(x))

#         x = x.transpose(1, 2)

#         x, _ = self.lstm7(self.drop7(x))
#         # x = self.linear7(x)

#         # x, _ = self.lstm8(self.drop8(x))
#         x = self.linear8(x)

#         return x

# def init_weights(m):
#     if type(m) == nn.Conv2d or type(m) == FullGatedConv2d:
#         nn.init.kaiming_uniform_(m.weight)
#     # nn.init.kaiming_uniform_(m.weight)


def create_model():
    model = CNNBLSTM()
    # model.apply(init_weights)
    return model


model_path = 'language_model/train.binary'
decoder = CTCBeamDecoder([*letters, '~'],
                         model_path=None,
						 alpha=0.01,
						 blank_id=len(letters), 
			 			 beam_width=100,
			 			 num_processes=n_cpus)
# decoder = CTCBeamDecoder([*letters, '~'],
                         # model_path=model_path,
                         # alpha=0.1,
						 # blank_id=len(letters), 
						 # beam_width=100,
						 # num_processes=n_cpus)


def get_prediction(act_model, test_images):
    act_model.eval()
    with torch.no_grad():
        output = F.softmax(act_model(test_images), dim=-1)

    beam_results, _, _, out_lens = decoder.decode(output)

    prediction = []
    for i in range(len(beam_results)):
        pred = "".join(letters[n] for n in beam_results[i][0][:out_lens[i][0]])

        prediction.append(pred)
    return prediction


def write_prediction(names_test, prediction, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for _, (name, line) in enumerate(zip(names_test, prediction)):
        with open(os.path.join(output_dir, name.replace('.jpg', '.txt')), 'w') as file:
            file.write(line)


def load_test_images(test_image_dir):
    test_images = []
    names_test = []
    for name in os.listdir(test_image_dir):
        img = Image.open(test_image_dir + '/' + name)
        img = process_image(img).unsqueeze(0)
        test_images.append(img)
        names_test.append(name)
    test_images = torch.cat(test_images, dim=0)
    return names_test, test_images


def main():
    test_image_dir = '/data'
    filepath = 'checkpoint/model.pth'
    pred_path = '/output'

    print('Creating model...', end=' ')
    act_model = create_model()
    print('Success')

    print(f'Loading weights from {filepath}...', end=' ')
    act_model.load_state_dict(torch.load(filepath))
    print('Success')

    print(f'Loading test images from {test_image_dir}...', end=' ')
    names_test, test_images = load_test_images(test_image_dir)
    print('Success')

    print('Running inference...')
    prediction = get_prediction(act_model, test_images)

    print('Writing predictions...')
    write_prediction(names_test, prediction, pred_path)
    
    
if __name__ == '__main__':
    main()