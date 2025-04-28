import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # change this as needed

use_cuda = torch.cuda.is_available()
seed = 2
os.environ['PYTHONHASHSEED'] = str(seed)


n_filts = 8
cosineLR = True         
n_channels = 3
n_labels = 1
epochs = 200
img_size = 256
print_frequency = 1
save_frequency = 100
vis_frequency = 50
early_stopping_patience = 50

pretrain = False


task_name = 'ISIC17_exp1'
# task_name = 'ISIC18_exp1'


#tmptest
learning_rate = 1e-5
batch_size = 16

# model_name = 'UNet_base'
# model_name = 'TransUnet'
# model_name = 'SwinUnet'
# model_name = 'UNetV2'
model_name = 'vmunet_v2' # VmCUnet
# model_name = 'DoubleUnet' 



test_session = "session_CIM"



## ISIC2017 data
train_dataset = '../data/ISIC2017/train'
val_dataset = '../data/ISIC2017/valid'
test_dataset = '../data/ISIC2017/test'

## ISIC2018 data    
# train_dataset = '../data/ISIC2018/train'
# val_dataset = '../data/ISIC2018/valid'
# test_dataset = '../data/ISIC2018/test'





session_name       = 'session_CIM' 
save_path          = task_name +'/'+ model_name +'/' + session_name + '/'
model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'




##########################################################################
# Trans configs
##########################################################################

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config

def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.pretrained_path = './nets/transunet/imagenet21k+imagenet2012_R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = 'softmax'

    return config

########################
#VMunet
########################
def get_vmamba_config():
    config = ml_collections.ConfigDict()
    
    # config.depths = [2, 2, 9, 2]
    # config.depths_decoder = [2, 9, 2, 2]
    
    # vm-unet 
    config.depths = [2, 2, 2, 2]
    config.depths_decoder = [2, 2, 2, 1]
    
    config.drop_path_rate = 0.2
    config.load_ckpt_path = './pre_trained_weights/vmamba_small_e238_ema.pth'
    config.deep_supervision = True
    return config
