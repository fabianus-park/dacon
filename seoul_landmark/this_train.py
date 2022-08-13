import datetime

import pandas as pd
import torch
import torch.optim as optim  # 최적화 알고리즘들이 포함힘
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import *
from preprocessing_data import *
from this_configuation import *


def train(model, optimizer, train_loader, vali_loader, scheduler, device): 
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y%m%d%H%M%S')

    model.to(device)
    n = len(train_loader)
    best_acc = 0
    
    for epoch in range(1,CFG["EPOCHS"]+1): #에포크 설정
        model.train() #모델 학습
        running_loss = 0.0
            
        for img, label in tqdm(iter(train_loader)):
            img, label = img.to(device), label.to(device) #배치 데이터
            optimizer.zero_grad() #배치마다 optimizer 초기화
        
            # Data -> Model -> Output
            logit = model(img) #예측값 산출
            loss = criterion(logit, label) #손실함수 계산
            
            # 역전파
            loss.backward() #손실함수 기준 역전파 
            optimizer.step() #가중치 최적화
            running_loss += loss.item()
              
        print('[%d] Train loss: %.10f' %(epoch, running_loss / len(train_loader)))
        
        if scheduler is not None:
            scheduler.step()
            
        #Validation set 평가
        model.eval() #evaluation 과정에서 사용하지 않아야 하는 layer들을 알아서 off 시키도록 하는 함수
        vali_loss = 0.0
        correct = 0
        with torch.no_grad(): #파라미터 업데이트 안하기 때문에 no_grad 사용
            for img, label in tqdm(iter(vali_loader)):
                img, label = img.to(device), label.to(device)

                logit = model(img)
                vali_loss += criterion(logit, label)
                pred = logit.argmax(dim=1, keepdim=True)  #11개의 class중 가장 값이 높은 것을 예측 label로 추출
                correct += pred.eq(label.view_as(pred)).sum().item() #예측값과 실제값이 맞으면 1 아니면 0으로 합산
        vali_acc = 100 * correct / len(vali_loader.dataset)
        print('Vail set: Loss: {:.4f}, Accuracy: {}/{} ( {:.0f}%)\n'.format(vali_loss / len(vali_loader), correct, len(vali_loader.dataset), 100 * correct / len(vali_loader.dataset)))
        
        #베스트 모델 저장
        if best_acc < vali_acc:
            best_acc = vali_acc
            torch.save(model.state_dict(), '.\\seoul_landmark\\best_model.'+nowDatetime+'.pth') #이 디렉토리에 best_model.pth을 저장
            print('Model Saved.')


if __name__ == "__main__":

    label_df = pd.read_csv('.\\seoul_landmark\\dataset\\train.csv')
    print( label_df.head() )
    
    all_img_path = []
    all_img_path = get_file_list('.\\seoul_landmark\\dataset\\train')
    print( all_img_path[:5] )
    
    label_df['file_path'] = all_img_path
    print( label_df.head() )
    
    all_label = []
    all_label.extend(label_df['label'])
    print( all_label[:5] )


    # Train : Validation = 0.75 : 0.25 Split
    train_len = int(len(all_img_path)*0.75)
    Vali_len = int(len(all_img_path)*0.25)

    train_img_path = all_img_path[:train_len]
    train_label = all_label[:train_len]

    vali_img_path = all_img_path[train_len:]
    vali_label = all_label[train_len:]

    print('train set 길이 : ', train_len)
    print('vaildation set 길이 : ', Vali_len)


    #CustomDataset class를 통하여 train dataset생성
    train_dataset = CustomDataset(train_img_path, train_label, train_mode=True, transforms=train_transform) 
    #만든 train dataset를 DataLoader에 넣어 batch 만들기
    train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)
    train_batches = len(train_loader)

    #vaildation 에서도 적용
    vali_dataset = CustomDataset(vali_img_path, vali_label, train_mode=True, transforms=test_transform)
    vali_loader = DataLoader(vali_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
    vali_batches = len(vali_loader)

    print('total train imgs :',train_len,'/ total train batches :', train_batches)
    print('total valid imgs :',Vali_len, '/ total valid batches :', vali_batches)



    # model = CNNclassification().to(device)

    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(params = model.parameters(), lr = CFG["LEARNING_RATE"])
    # scheduler = None

    # train(model, optimizer, train_loader, vali_loader, scheduler, device)


    # load resnet18 with the pre-trained weights
    # this_model = torchvision.models.resnet18(pretrained=True)
    this_model = torchvision.models.resnet152(pretrained=True)

    num_classes = 10
    num_ftrs = this_model.fc.in_features
    this_model.fc = nn.Linear(num_ftrs, num_classes)

    this_model.to(device)


    # get the model summary
    from torchsummary import summary
    summary(this_model, input_size=(3, 224, 224), device=device.type)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params = this_model.parameters(), lr = CFG["LEARNING_RATE"])
    scheduler = None

    train(this_model, optimizer, train_loader, vali_loader, scheduler, device)
