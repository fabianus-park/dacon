import os
import torch
import torch.nn as nn
import random
import numpy as np

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
# os.environ["CUDA_VISIBLE_DEVICES"]= "2"  # Set the GPU 2 to use, 멀티 gpu

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#GPU 체크 및 할당
if torch.cuda.is_available():    
    #device = torch.device("cuda:0")
#    device = torch.device('cuda')
    print('Device:', device)
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
#    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')


#하이퍼 파라미터
CFG = {
    'IMG_SIZE':128, #이미지 사이즈
    'EPOCHS':50, #에포크
    'LEARNING_RATE':2e-2, #학습률
    'BATCH_SIZE':12, #배치사이즈
#    'SEED':41, #시드
    'SEED':99, #시드
}

# Seed 고정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED'])

dataset_path = '.\\seoul_landmark\\dataset\\'