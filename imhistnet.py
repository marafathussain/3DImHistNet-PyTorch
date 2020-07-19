import torch
import torch.nn as nn

# Remember that you have add the following lines in your main training code after calling the model:
'''
model = imhistnet_3d().to(device)
for name, param in model.named_parameters():
    if name in ['conv1.weight', 'conv2.bias']:
        param.required_grad = False
'''

# PLease ignore the following commented portion
'''
class imhistnet_3d(nn.Module):
    def __init__(self,
                 bins=32,
                 no_classes=2,
                 pool_kernel=32,
                 pool_stride=32):
        super(imhistnet_3d, self).__init__()

        self.conv1 = nn.Conv3d(1, bins, 1, 1)
        self.conv2 = nn.Conv3d(bins, bins, 1, 1, groups=bins)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool3d(pool_kernel, pool_stride)
        self.fc1 = nn.Linear(bins*4*4*4, 1024)
        self.fc2 = nn.Linear(1024, no_classes)

        initialize_params(self)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.abs(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
    
def initialize_params(module):

    for name, m in module.named_modules():
        if isinstance(m, nn.Conv3d) and name=='conv1':
            nn.init.constant_(m.weight, 1.0)
            nn.init.xavier_normal_(m.bias)

        elif isinstance(m, nn.Conv3d) and name=='conv2':
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 1.0)

        else:
            nn.init.xavier_normal_(m.weight)
            nn.init.xavier_normal_(m.bias)
'''

class imhistnet_3d(nn.Module):
    def __init__(self,
                 bins=16,
                 no_classes=2,
                 pool_kernel=32,
                 pool_stride=32):
        super(imhistnet_3d, self).__init__()

        self.conv1 = nn.Conv3d(1, bins, 1, 1)
        nn.init.constant_(self.conv1.weight, 1.0)
        
        
        self.conv2 = nn.Conv3d(bins, bins, 1, 1, groups=bins)
        nn.init.constant_(self.conv2.bias, 1.0)
        
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool3d(pool_kernel, pool_stride)
        self.fc1 = nn.Linear(bins*4*4*4, 1024) # calculating the first part is required based on the bins and pool size
        self.fc2 = nn.Linear(1024, no_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = torch.abs(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x