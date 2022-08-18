import torchvision
import torch.nn as nn

# this_model = torchvision.models.efficientnet_b3(pretrained=False)
# print(this_model.classifier)
# #print(this_model..in_features)
# this_model.fc = nn.Linear(1000, 10)
# print(this_model.feature)



this_model = torchvision.models.resnet152(pretrained=True)

### print(this_model.fc)
### print(this_model.fc.in_features)
### num_classes = 10
### num_ftrs = this_model.fc.in_features
### print(num_ftrs, num_classes)
### this_model.fc = nn.Linear(num_ftrs, num_classes)
### 
### # torchvision.models.googlenet()
### 

