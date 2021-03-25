from typing import OrderedDict
import torch
from torch.nn.modules.batchnorm import BatchNorm2d
import torch.nn as nn
import torch.optim as optim
import os
import torch.onnx
import functools


# Pix2Pix model class [256 based]
class Pix2PixModel():
    def __init__(self, ckpt_dir, model_name,
                is_train=True, n_epochs=100, 
                n_epochs_decay=100):
        super(Pix2PixModel, self).__init__()
        self.isTrain = is_train
        # self.training = self.isTrain
        self.save_dir = ckpt_dir
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        self.optimizers = []
        self.epoch_count = 1
        self.n_epochs = n_epochs
        self.n_epochs_decay = n_epochs_decay
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        # Define the Generator
        self.netG = Generator(3, 3, 8, ngf=64, norm_layer=norm_layer, use_dropout=False)
        self.netG = init_net(self.netG)

        # Define the Discriminator if training
        if self.isTrain:
            self.criterionLoss = nn.L1Loss()
            #self.netD = Discriminator(6, ndf=64, n_layers=3, norm_layer=norm_layer)
            self.netD = Discriminator(6)
            self.netD = init_net(self.netD)
        
        if self.isTrain:
            self.criterionGAN = GANLoss().to(self.device)
            self.criterionL1 = nn.L1Loss()
            self.optimizer_G = optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
            self.optimizer_D = optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        
    def set_input(self, input, single=False):
        if single:
            self.real_A = input.to(self.device)
            self.real_B = input.to(self.device)
        else:
            self.real_A = input['A'].to(self.device)
            self.real_B = input['B'].to(self.device)
            self.image_paths = input['A_paths']
        
    def forward(self):
        self.fake_B = self.netG(self.real_A)
    
    def backward_D(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
    
    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * 100.0

        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
    
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
    
    def get_scheduler(self, optimizer):
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + self.epoch_count - self.n_epochs) / float(self.n_epochs_decay + 1)
            return lr_l
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        return scheduler

    def setup(self):
        if self.isTrain:
            self.schedulers = [self.get_scheduler(optimizer) for optimizer in self.optimizers]
        
        self.print_networks()
    
    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        with torch.no_grad():
            self.forward()

    def get_image_paths(self):
        return self.image_paths

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def save_network(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                torch.save(net.cpu().state_dict(), save_path)
                net.to(self.device)
    
    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, nn.DataParallel):
                    net = net.module
                print('Loading the model from %s' % load_path)

                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                
                for key in list(state_dict.keys()):
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
    
    # def train(self, is_training):
    #     pass
    
class UnetBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                submodule=None, outermost=False, innermost=False,
                norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                            stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)

        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4,
                                        stride=2, padding=1, bias=use_bias)
            
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)

            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Generator, self).__init__()
        unet_block = UnetBlock(ngf * 8, ngf * 8, input_nc=None,
                                submodule=None, norm_layer=norm_layer,
                                innermost=True)
        
        for i in range(num_downs - 5):
            unet_block = UnetBlock(ngf * 8, ngf * 8, input_nc=None, 
                                    submodule=unet_block, norm_layer=norm_layer,
                                    use_dropout=use_dropout)
        
        unet_block = UnetBlock(ngf * 4, ngf * 8, input_nc=None,
                                submodule=unet_block, norm_layer=norm_layer)
        
        unet_block = UnetBlock(ngf * 2, ngf * 4, input_nc=None,
                                submodule=unet_block, norm_layer=norm_layer)
        
        unet_block = UnetBlock(ngf, ngf * 2, input_nc=None,
                                submodule=unet_block, norm_layer=norm_layer)

        self.model = UnetBlock(output_nc, ngf, input_nc=input_nc, 
                                submodule=unet_block, 
                                outermost=True, norm_layer=norm_layer)
        

    def forward(self, input):
        return self.model(input)

class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), 
                    nn.LeakyReLU(0.2, True)]
        
        nf_mult = 1
        nf_mult_prev = 1
        
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)

            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                        kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                        norm_layer(ndf * nf_mult),
                        nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8) # type: ignore

        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1,
                        padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)
    
    def forward(self, input):
        return self.model(input)

class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.gan_mode = 'vanilla'
        self.loss = nn.BCEWithLogitsLoss()
    
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    
    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)

        return loss

def init_net(net):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net.to(device)

        def init_weights(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        
        net.apply(init_weights)
        return net
