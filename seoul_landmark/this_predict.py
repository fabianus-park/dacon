import datetime

import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models import *
from preprocessing_data import *
from this_configuation import CFG, device


def predict(model, test_loader, device):
    model.eval()
    model_pred = []
    with torch.no_grad():
        for img in tqdm(iter(test_loader)):
            img = img.to(device)

            pred_logit = model(img)
            pred_logit = pred_logit.argmax(dim=1, keepdim=True).squeeze(1)

            model_pred.extend(pred_logit.tolist())
    return model_pred


if __name__ == "__main__":
    
    test_img_path = get_file_list('.\\seoul_landmark\\dataset\\test')

    test_dataset = CustomDataset(test_img_path, None, train_mode=False, transforms=test_transform)
    test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    # Validation Accuracy가 가장 뛰어난 모델을 불러옵니다.
    # checkpoint = torch.load('.\\seoul_landmark\\best_model.20220812185302.pth')
    # this_model = CNNclassification().to(device)
    # this_model.load_state_dict(checkpoint)

    checkpoint = torch.load('.\\seoul_landmark\\best_model.20220813134818.pth')
    #this_model = CNNclassification().to(device)

    #this_model = torchvision.models.resnet18(pretrained=True)
    this_model = torchvision.models.resnet152(pretrained=True)
    num_classes = 10
    num_ftrs = this_model.fc.in_features
    this_model.fc = nn.Linear(num_ftrs, num_classes)
    this_model.to(device)

    this_model.load_state_dict(checkpoint)

    # Inference
    preds = predict(this_model, test_loader, device)
    print( preds[0:5] )

    submission = pd.read_csv('.\\seoul_landmark\\dataset\\sample_submission.csv')
    submission['label'] = preds

    print( submission.head() )

    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y%m%d%H%M%S')

    submission.to_csv('.\\seoul_landmark\\dataset\\submission' + nowDatetime + '.csv', index=False)
