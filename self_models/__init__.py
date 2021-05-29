from .vgg_spiking import *
from .vgg_hrank import *
from .vgg import *
from .resnet import *
from .resnet_spiking_base import *
from .resnet_spiking_batch_norm_type7 import *
from .resnet_spiking_se import *
from .vgg_spiking_imagenet import *
from .resnet_spiking_imagenet import *
from .densenet_spiking import *
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .resnetv2 import ResNet50


model_dict = {
    # 'resnet8': resnet8,
    # 'resnet14': resnet14,
    # 'resnet20': resnet20,
    # 'resnet32': resnet32,
    # 'resnet44': resnet44,
    # 'resnet56': resnet56,
    # 'resnet110': resnet110,
    # 'resnet8x4': resnet8x4,
    # 'resnet32x4': resnet32x4,
    'ResNet50': ResNet50,
    'vgg16_hrank': vgg_16_hrank,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    # 'vgg11': vgg11_bn,
    # 'vgg13': vgg13_bn,
    # 'vgg16': vgg16_bn,
    # 'vgg19': vgg19_bn,
    # 'MobileNetV2': mobile_half,
    # 'ShuffleV1': ShuffleV1,
    # 'ShuffleV2': ShuffleV2,
    # 'resnet14x05': resnet14x05,
    # 'resnet20x05': resnet20x05,
    # 'resnet20x0375': resnet20x0375,
}