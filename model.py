import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    @staticmethod
    def gram_matrix(input):
        #(a, b, c, d) := (batch, num feats, h, w)
        a, b, c, d = input.size()
        features = input.view(a * b * c, d)
        G = torch.mm(features, features.t())
        return G.div(d)

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = StyleLoss.gram_matrix(target_feature).detach()

    def forward(self, input):
        G = StyleLoss.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


class RandomCNN(nn.Module):
    def __init__(self, out_channels=32, kernel=(1, 1)):
        super(RandomCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, out_channels, kernel_size=kernel, bias=False)
        #self.LeakyReLU = nn.LeakyReLU(0.2)

        # Set the conv parameters to be constant
        weight = torch.randn(self.conv1.weight.data.shape)
        # bias = torch.zeros(self.conv1.bias.data.shape)

        self.conv1.weight = torch.nn.Parameter(weight, requires_grad=False)
        # self.conv1.bias = torch.nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        out = self.conv1(x)
        #out = self.LeakyReLU(out)
        return out


def get_style_model_and_losses(cnn, content, style):
    content_losses = []
    style_losses = []

    #model = nn.Sequential(normalization) #???
    model = nn.Sequential()
    model.eval()

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.LeakyReLU):
            name = 'relu_{}'.format(i)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        # add content loss:
        if 'conv' in name:
            target = model(content).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        # add style loss:
        if 'conv' in name:
            target_feature = model(style).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)


    return model, content_losses, style_losses


def run_transfer(cnn, content_spectrum, style_spectrum,
                 num_steps=300,
                 content_weight=1, style_weight=1e4):

    print('Building the style transfer model..')
    print_freq = int(num_steps // 20)

    content_length = content_spectrum.shape[1]
    style_length = style_spectrum.shape[1]

    if style_length > content_length:
        style_spectrum = style_spectrum[:, :content_length]
    else:
        pad = [(0, 0), (0, content_length - style_length)]
        style_spectrum = np.pad(style_spectrum, pad_width=pad)

    content = torch.from_numpy(content_spectrum)[None, None, :, :].to(device)
    style = torch.from_numpy(style_spectrum)[None, None, :, :].to(device)

    content.requires_grad_(False)
    style.requires_grad_(False)

    result = (torch.randn(content.data.size()) * 1e-3).to(device)

    cnn = cnn.to(device)

    model, content_losses, style_losses = get_style_model_and_losses(cnn, content, style)

    result.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)

    optimizer = optim.LBFGS([result])
    #optimizer = optim.Adam([result])

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            # ???
            #with torch.no_grad():
            #result.clamp_(0, 1)

            optimizer.zero_grad()
            model(result)
            content_score = 0
            style_score = 0

            for cl in content_losses:
                content_score += cl.loss
            for sl in style_losses:
                style_score += sl.loss

            content_score *= content_weight
            style_score *= style_weight

            loss = content_score + style_score
            loss.backward()

            run[0] += 1
            if run[0] % print_freq == 0:
                print('{}: Content Loss: {:.4f} Style Loss : {:.4f}'.format(run[0], content_score.item(), style_score.item()))

            return content_score + style_score

        optimizer.step(closure)
        #closure()
        #optimizer.step()

    # ???
    #with torch.no_grad():
    #result.clamp_(0, 1)

    return result
