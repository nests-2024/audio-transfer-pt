import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)
    
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
    def __init__(self, out_channels=32, kernel=(3, 1)):
        super(RandomCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, out_channels, kernel_size=kernel)
        self.LeakyReLU = nn.LeakyReLU(0.2)

        # Set the conv parameters to be constant
        weight = torch.randn(self.conv1.weight.data.shape)
        bias = torch.zeros(self.conv1.bias.data.shape)

        self.conv1.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.conv1.bias = torch.nn.Parameter(bias, requires_grad=False)

    def forward(self, x_delta):
        out = self.LeakyReLU(self.conv1(x_delta))
        return out


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img])
    return optimizer


def get_style_model_and_losses(cnn, style, content):
    content_losses = []
    style_losses = []

    #model = nn.Sequential(normalization) #???
    model = nn.Sequential()

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

    return model, style_losses, content_losses


def run_style_transfer(cnn,
                       content_spectrum, style_spectrum, result, num_steps=300,
                       style_weight=1e6, content_weight=1):

    print('Building the style transfer model..')

    content = torch.from_numpy(content_spectrum)[None, None, :, :]
    style = torch.from_numpy(style_spectrum)[None, None, :, :]

    result = torch.from_numpy(content_spectrum)[None, None, :, :]
    result = torch.randn(content.data.size())

    model, style_losses, content_losses = get_style_model_and_losses(cnn, style, content)

    result.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(result)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            # ???
            #with torch.no_grad():
            #result.clamp_(0, 1)

            optimizer.zero_grad()
            model(result)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # ???
    #with torch.no_grad():
    #result.clamp_(0, 1)

    return result
