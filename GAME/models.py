from pydoc import describe
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import torchvision
import GAME.datasets as datasets
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from rich.progress import track
import math
irange = range
import numpy as np

# Target Model definition
class MNIST_target_net(nn.Module):
    def __init__(self):
        super(MNIST_target_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)

        self.fc1 = nn.Linear(64*4*4, 200)
        self.fc2 = nn.Linear(200, 200)
        self.logits = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64*4*4)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = self.logits(x)
        return x



class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class LeNet(nn.Module):
    def __init__(self,n_channels=1,n_outputs=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_outputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x= F.softmax(x)
        return x



class HalfLeNet(nn.Module):
    # def __init__(self, name, n_outputs):
    def __init__(self,n_channels=1,n_outputs=10):
        super().__init__()

        self.conv1 = nn.Conv2d(
                in_channels = n_channels,
                out_channels = 3,
                kernel_size = 5,
                stride = 1,
                padding = 0,
            )
        self.conv1.bias.data.normal_(0, 0.1)
        self.conv1.bias.data.fill_(0)

        self.max_pool_1 = nn.MaxPool2d(kernel_size = 2)

        self.conv2 = nn.Conv2d(
                in_channels = 3,
                out_channels = 8,
                kernel_size = 5,
                stride = 1,
                padding = 0,
            )
        self.conv2.bias.data.normal_(0, 0.1)
        self.conv2.bias.data.fill_(0)

        self.max_pool_2 = nn.MaxPool2d(kernel_size = 2)

        self.fc1 = nn.Linear(8 * 5 * 5, 120)
        self.fc1.bias.data.normal_(0, 0.1)
        self.fc1.bias.data.fill_(0)

        self.fc2 = nn.Linear(120, 84)
        self.fc2.bias.data.normal_(0, 0.1)
        self.fc2.bias.data.fill_(0)

        self.fc3 = nn.Linear(84, n_outputs)
        self.fc3.bias.data.normal_(0, 0.1)
        self.fc3.bias.data.fill_(0)


    def forward(self, input):
        x = torch.relu(self.conv1(input))
        x = self.max_pool_1(x)
        x = torch.relu(self.conv2(x))
        x = self.max_pool_2(x)
        x = x.view(-1, 8 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x



class ResNet18(nn.Module):
    def __init__(self, name="ResNet18", n_channels=1, n_outputs=10):
        super().__init__()
        self.name = name
        self.n_outputs = n_outputs

        self.model = torchvision.models.resnet18(pretrained = False)
        # 输入修改为单通道
        self.model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, n_outputs)
        #input_size = 224

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs

class ResNet34(nn.Module):
    def __init__(self, name="ResNet34", n_channels=1, n_outputs=10):
        super().__init__()
        self.name = name
        self.n_outputs = n_outputs

        self.model = torchvision.models.resnet34(pretrained = False)
        # 输入修改为单通道
        self.model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, n_outputs)
        #input_size = 224

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs

class ResNet101(nn.Module):
    def __init__(self, name="ResNet101", n_channels=1, n_outputs=10):
        super().__init__()
        self.name = name
        self.n_outputs = n_outputs

        self.model = torchvision.models.resnet101(pretrained = False)
        # 输入修改为单通道
        self.model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, n_outputs)
        #input_size = 224

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs

class VGG16(nn.Module):
    def __init__(self, name="VGG16", n_channels = 1, n_outputs = 10):
        super(VGG16, self).__init__()
        self.name = name

        self.n_channels = n_channels
        self.features = self._make_layers([
            64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
            512, 512, 512, 'M', 512, 512, 512, 'M'
        ])
        if n_outputs == 40:
            self.classifier = nn.Linear(32768, n_outputs)
        else:
            self.classifier = nn.Linear(512, n_outputs)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = torch.nn.functional.dropout(out, p = .5, training = self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, configuration):
        layers = []
        in_channels = self.n_channels
        for x in configuration:
            if x == 'M':
                layers += [ nn.MaxPool2d(kernel_size = 2, stride = 2) ]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size = 3, padding = 1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size = 1, stride = 1)]
        return nn.Sequential(*layers)

class VGG13(nn.Module):
    def __init__(self, name="VGG13", n_channels = 1, n_outputs = 10):
        super(VGG13, self).__init__()
        self.name = name

        self.n_channels = n_channels
        self.features = self._make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'])
        self.classifier = nn.Linear(512, n_outputs)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = torch.nn.functional.dropout(out, p = .5, training = self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, configuration):
        layers = []
        in_channels = self.n_channels
        for x in configuration:
            if x == 'M':
                layers += [ nn.MaxPool2d(kernel_size = 2, stride = 2) ]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size = 3, padding = 1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size = 1, stride = 1)]
        return nn.Sequential(*layers)

class VGG19(nn.Module):
    def __init__(self, name="VGG19", n_channels = 1, n_outputs = 10):
        super(VGG19, self).__init__()
        self.name = name

        self.n_channels = n_channels
        self.features = self._make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'])
        self.classifier = nn.Linear(512, n_outputs)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = torch.nn.functional.dropout(out, p = .5, training = self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, configuration):
        layers = []
        in_channels = self.n_channels
        for x in configuration:
            if x == 'M':
                layers += [ nn.MaxPool2d(kernel_size = 2, stride = 2) ]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size = 3, padding = 1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size = 1, stride = 1)]
        return nn.Sequential(*layers)

class Alexnet(nn.Module):
    def __init__(self, name='alexnet',n_channels=3, n_outputs=10):
        super(Alexnet, self).__init__()

        self.name = name
        self.num_classes = n_outputs

        self.conv1 = nn.Conv2d(n_channels, 48, 5, stride=1, padding=2)
        self.conv1.bias.data.normal_(0, 0.01)
        self.conv1.bias.data.fill_(0) 
        
        self.relu = nn.ReLU()        
        self.lrn = nn.LocalResponseNorm(2)        
        self.pad = nn.MaxPool2d(3, stride=2)
        
        self.batch_norm1 = nn.BatchNorm2d(48, eps=0.001)
        
        self.conv2 = nn.Conv2d(48, 128, 5, stride=1, padding=2)
        self.conv2.bias.data.normal_(0, 0.01)
        self.conv2.bias.data.fill_(1.0)  
        
        self.batch_norm2 = nn.BatchNorm2d(128, eps=0.001)
        
        self.conv3 = nn.Conv2d(128, 192, 3, stride=1, padding=1)
        self.conv3.bias.data.normal_(0, 0.01)
        self.conv3.bias.data.fill_(0)  
        
        self.batch_norm3 = nn.BatchNorm2d(192, eps=0.001)
        
        self.conv4 = nn.Conv2d(192, 192, 3, stride=1, padding=1)
        self.conv4.bias.data.normal_(0, 0.01)
        self.conv4.bias.data.fill_(1.0)  
        
        self.batch_norm4 = nn.BatchNorm2d(192, eps=0.001)
        
        self.conv5 = nn.Conv2d(192, 128, 3, stride=1, padding=1)
        self.conv5.bias.data.normal_(0, 0.01)
        self.conv5.bias.data.fill_(1.0)  
        
        self.batch_norm5 = nn.BatchNorm2d(128, eps=0.001)
        
        self.fc1 = nn.Linear(1152,512)
        self.fc1.bias.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0) 
        
        self.drop = nn.Dropout(p=0.5)
        
        self.batch_norm6 = nn.BatchNorm1d(512, eps=0.001)
        
        self.fc2 = nn.Linear(512,256)
        self.fc2.bias.data.normal_(0, 0.01)
        self.fc2.bias.data.fill_(0) 
        
        self.batch_norm7 = nn.BatchNorm1d(256, eps=0.001)
        
        self.fc3 = nn.Linear(256,self.num_classes)
        self.fc3.bias.data.normal_(0, 0.01)
        self.fc3.bias.data.fill_(0) 
        
        self.soft = nn.Softmax()
        
    def forward(self, x):
        layer1 = self.batch_norm1(self.pad(self.lrn(self.relu(self.conv1(x)))))
        layer2 = self.batch_norm2(self.pad(self.lrn(self.relu(self.conv2(layer1)))))
        layer3 = self.batch_norm3(self.relu(self.conv3(layer2)))
        layer4 = self.batch_norm4(self.relu(self.conv4(layer3)))
        layer5 = self.batch_norm5(self.pad(self.relu(self.conv5(layer4))))
        flatten = layer5.view(-1, 128*3*3)
        fully1 = self.relu(self.fc1(flatten))
        fully1 = self.batch_norm6(self.drop(fully1))
        fully2 = self.relu(self.fc2(fully1))
        fully2 = self.batch_norm7(self.drop(fully2))
        logits = self.fc3(fully2)
        #softmax_val = self.soft(logits)

        return logits

class HalfAlexnet(nn.Module):
    def __init__(self, name='half-alexnet',n_channels=3, n_outputs=10):
        super(HalfAlexnet, self).__init__()

        self.name = name
        self.num_classes = n_outputs

        self.conv1 = nn.Conv2d(n_channels, 24, 5, stride=1, padding=2)
        self.conv1.bias.data.normal_(0, 0.01)
        self.conv1.bias.data.fill_(0) 
        
        self.relu = nn.ReLU()        
        self.lrn = nn.LocalResponseNorm(2)        
        self.pad = nn.MaxPool2d(3, stride=2)
        
        self.batch_norm1 = nn.BatchNorm2d(24, eps=0.001)
        
        self.conv2 = nn.Conv2d(24, 64, 5, stride=1, padding=2)
        self.conv2.bias.data.normal_(0, 0.01)
        self.conv2.bias.data.fill_(1.0)  
        
        self.batch_norm2 = nn.BatchNorm2d(64, eps=0.001)
        
        self.conv3 = nn.Conv2d(64, 96, 3, stride=1, padding=1)
        self.conv3.bias.data.normal_(0, 0.01)
        self.conv3.bias.data.fill_(0)  
        
        self.batch_norm3 = nn.BatchNorm2d(96, eps=0.001)
        
        self.conv4 = nn.Conv2d(96, 96, 3, stride=1, padding=1)
        self.conv4.bias.data.normal_(0, 0.01)
        self.conv4.bias.data.fill_(1.0)  
        
        self.batch_norm4 = nn.BatchNorm2d(96, eps=0.001)
        
        self.conv5 = nn.Conv2d(96, 64, 3, stride=1, padding=1)
        self.conv5.bias.data.normal_(0, 0.01)
        self.conv5.bias.data.fill_(1.0)  
        
        self.batch_norm5 = nn.BatchNorm2d(64, eps=0.001)
        
        self.fc1 = nn.Linear(576,256)
        self.fc1.bias.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0) 
        
        self.drop = nn.Dropout(p=0.5)
        
        self.batch_norm6 = nn.BatchNorm1d(256, eps=0.001)
        
        self.fc2 = nn.Linear(256,128)
        self.fc2.bias.data.normal_(0, 0.01)
        self.fc2.bias.data.fill_(0) 
        
        self.batch_norm7 = nn.BatchNorm1d(128, eps=0.001)
        
        self.fc3 = nn.Linear(128,self.num_classes)
        self.fc3.bias.data.normal_(0, 0.01)
        self.fc3.bias.data.fill_(0) 
        
        self.soft = nn.Softmax()
        
    def forward(self, x):
        layer1 = self.batch_norm1(self.pad(self.lrn(self.relu(self.conv1(x)))))
        layer2 = self.batch_norm2(self.pad(self.lrn(self.relu(self.conv2(layer1)))))
        layer3 = self.batch_norm3(self.relu(self.conv3(layer2)))
        layer4 = self.batch_norm4(self.relu(self.conv4(layer3)))
        layer5 = self.batch_norm5(self.pad(self.relu(self.conv5(layer4))))
        flatten = layer5.view(-1, 64*3*3)
        fully1 = self.relu(self.fc1(flatten))
        fully1 = self.batch_norm6(self.drop(fully1))
        fully2 = self.relu(self.fc2(fully1))
        fully2 = self.batch_norm7(self.drop(fully2))
        logits = self.fc3(fully2)
        #softmax_val = self.soft(logits)

        return logits



class Generator2(nn.Module):
    def __init__(self, n_classes, img_size, channels, opt, latent_dim=100):
        super(Generator2, self).__init__()
        self.n_classes = n_classes
        self.opt = opt

        num_embeddings = n_classes
        self.label_emb = nn.Embedding(num_embeddings, latent_dim)

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 512 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def label2int(self,label):
        # if len(label.shape) == 1:
        #     return torch.Tensor([int("".join([str(int(i)) for i in label]),2)])
        # else:
        w = torch.Tensor([[2**(self.n_classes-1-i)] for i in range(self.n_classes)]).type(torch.float32).cuda()
        res = label.mm(w)
        res = res.squeeze(1).long()
        return res

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img



class Generator(nn.Module):
    def __init__(self, n_classes, img_size, channels, opt, latent_dim=100):
        super(Generator, self).__init__()
        self.n_classes = n_classes
        self.opt = opt

        num_embeddings = n_classes
        self.label_emb = nn.Embedding(num_embeddings, latent_dim)

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def label2int(self,label):
        # if len(label.shape) == 1:
        #     return torch.Tensor([int("".join([str(int(i)) for i in label]),2)])
        # else:
        w = torch.Tensor([[2**(self.n_classes-1-i)] for i in range(self.n_classes)]).type(torch.float32).cuda()
        res = label.mm(w)
        res = res.squeeze(1).long()
        return res

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, n_classes, img_size, channels):
        super(Discriminator, self).__init__()
        self.n_class = n_classes

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 5

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(256 * ds_size ** 2, 1), nn.Sigmoid())
        #self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, n_classes), nn.Softmax())
        self.aux_layer = nn.Sequential(nn.Linear(256 * ds_size ** 2, n_classes+1))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)

        validity = self.adv_layer(out).view(-1)
        label = self.aux_layer(out).view(-1, self.n_class+1)

        return validity, label




def CIFAR10():
    return VGG16(n_channels = 3)

def GTSRB():
    return VGG16(n_channels=3, n_outputs=43)

def MNIST():
    return LeNet()

def FashionMNIST():
    return LeNet()


class ACGAN():
    def __init__(self, opt, id_G=1):
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_output = datasets.datasets_dict[self.opt.proxyset]["n_outputs"]
        self.n_channels = datasets.datasets_dict[self.opt.proxyset]["n_channels"]
        self.image_size = datasets.datasets_dict[self.opt.proxyset]["img_size"]
        self.id_G = id_G

        self.sampler_weights = torch.ones(self.n_output)

        if id_G == 1:
            self.generator = Generator(self.n_output,self.image_size,self.n_channels,self.opt,latent_dim=10*self.n_output).to(self.device)
        elif id_G == 2:
            self.generator = Generator2(self.n_output,self.image_size,self.n_channels,self.opt,latent_dim=10*self.n_output).to(self.device)
        self.discriminator = Discriminator(self.n_output,self.image_size,self.n_channels).to(self.device)

        self.generator.apply(self._weights_init_)
        self.discriminator.apply(self._weights_init_)

        self.optimizer_G = optim.Adam(self.generator.parameters(),0.0002, betas=(0.5,0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), 0.0002,betas=(0.5,0.999))

        self.validity_loss = nn.BCELoss()

        self.real_labels = 0.7 + 0.5 * torch.rand(self.n_output, device = self.device)
        self.fake_labels = 0.3 * torch.rand(self.n_output, device = self.device)
    
    def _weights_init_(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def save_samples(self,tags):
        batch_size = 36
        noise = torch.randn(batch_size, 10*self.n_output, device = self.device)     
        # sample_labels = torch.randint(0, self.n_output, (batch_size,),device = self.device, dtype = torch.long)
        sample_labels = torch.cat([torch.ones(6)*i*3 for i in range(6)]).long().to(self.device)
        imgs = self.generator(noise,sample_labels)

        file_name = os.path.join(self.opt.img_dir,self.opt.testname,tags+".png")
        save_image(imgs,file_name,normalize=True)
        # print("Images save success:",file_name)



    def train(self,trainloader:DataLoader,n_epoch=10,load=True):
        file_G = "generator%s-%s-%s.pth"%(self.id_G,self.opt.proxyset,n_epoch)

        if load and os.path.exists(os.path.join(self.opt.model_dir,file_G)):
            generator_load = torch.load(os.path.join(self.opt.model_dir,file_G))
            self.generator.load_state_dict(generator_load)
            print("ACGAN load success!")
            return 0

        
        self.generator.train()
        self.discriminator.train()
        print("Start to train ACGAN!")
        
        cuda = True if torch.cuda.is_available() else False

        for epoch in track(range(1, n_epoch+1),description="Train ACGAN:"):
            for i, (imgs, labels) in enumerate(trainloader,0):

                batch_size = imgs.size(0)

                if cuda:
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                else:
                    imgs = Variable(imgs)
                    labels = Variable(labels)
                
                real_label = self.real_labels[i % self.n_output]
                fake_label = self.fake_labels[i % self.n_output]

                fake_class_labels = self.n_output*torch.ones((batch_size,),dtype = torch.long,device = self.device)
                
                if i % 25 == 0:
                    real_label, fake_label = fake_label, real_label
               
                # -------------------------
                #         discriminator
                # -------------------------

                self.optimizer_D.zero_grad()
                # real image
                validity_label = torch.full((batch_size,),real_label, device = self.device)

                pvalidity, plabels = self.discriminator(imgs)

                errD_real_val = self.validity_loss(pvalidity, validity_label)
                errD_real_label = F.cross_entropy(plabels,labels)
                
                errD_real = errD_real_val + errD_real_label
                errD_real.backward()
                
                D_x = pvalidity.mean().item()

                # fake image
                noise = torch.randn(batch_size, 10*self.n_output, device = self.device)
                
                sample_labels = torch.randint(0, self.n_output, (batch_size,),device = self.device, dtype = torch.long)

                fakes = self.generator(noise,sample_labels)

                validity_label.fill_(fake_label)

                pvalidity, plabels = self.discriminator(fakes.detach())       
                
                errD_fake_val = self.validity_loss(pvalidity, validity_label)
                errD_fake_label = F.cross_entropy(plabels, fake_class_labels)
                
                errD_fake = errD_fake_val + errD_fake_label
                errD_fake.backward()
                
                D_G_z1 = pvalidity.mean().item()

                errD = errD_real + errD_fake
        
                self.optimizer_D.step()                
             
                # -------------------------
                #         generator
                # -------------------------
                self.optimizer_G.zero_grad()

                noise = torch.randn(batch_size,10*self.n_output,device = self.device)
                sample_labels = torch.randint(0,self.n_output,(batch_size,),device = self.device, dtype = torch.long)
                
                validity_label.fill_(1)
                
                fakes = self.generator(noise,sample_labels)
                pvalidity,plabels = self.discriminator(fakes)
                
                errG_val = self.validity_loss(pvalidity, validity_label)

                errG_label = F.cross_entropy(plabels, sample_labels)
                
                errG = errG_val + errG_label
                errG.backward()
                D_G_z2 = pvalidity.mean().item()
                
                self.optimizer_G.step()

                # print("  [ACGAN] [Epoch:{}/{}] [Batch:{}/{}] D_x: [{:.4f}] D_G: [{:.4f}/{:.4f}] G_loss: [{:.4f}] D_loss: [{:.4f}] D_label: [{:.4f}] ".format(epoch,n_epoch, i, len(trainloader),D_x, D_G_z1,D_G_z2,errG,errD,errD_real_label + errD_fake_label + errG_label))
            if epoch % 40 == 0:
                self.save_samples('Train-ACGAN-epoch%d'%(epoch))

        torch.save(self.generator.state_dict(),os.path.join(self.opt.model_dir,file_G))
        print("ACGAN train finished! New generator has been saved: %s"%(file_G))

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new_full((3, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=6, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)
