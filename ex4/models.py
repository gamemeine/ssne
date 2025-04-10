import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicNet(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class SimpleResNet(nn.Module):
    def __init__(self, num_classes=50):
        super(SimpleResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer0 = self._make_layer(64, blocks=2, stride=1)   # Keeps resolution: 64x64
        self.layer1 = self._make_layer(128, blocks=2, stride=2)  # Downsamples: 64 -> 32
        self.layer2 = self._make_layer(256, blocks=2, stride=2)  # Downsamples: 32 -> 16
        self.layer3 = self._make_layer(512, blocks=2, stride=2)  # Downsamples: 16 -> 8
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(ResidualBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(ResidualBlock(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)         # (B, 64, 64, 64)
        x = self.layer0(x)        # (B, 64, 64, 64)
        x = self.layer1(x)        # (B, 128, 32, 32)
        x = self.layer2(x)        # (B, 256, 16, 16)
        x = self.layer3(x)        # (B, 512, 8, 8)
        x = self.avgpool(x)       # (B, 512, 1, 1)
        x = torch.flatten(x, 1)   # (B, 512)
        x = self.fc(x)            # (B, num_classes)
        return x
    

class FourBlockCNN(nn.Module):
    def __init__(self, num_classes=50):
        super(FourBlockCNN, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.block1 = self.make_block(32, 32, num_convs=4, stride_down=2, dropout_rate=0.3)  # 64->32 spatial dims
        self.block2 = self.make_block(32, 64, num_convs=4, stride_down=2, dropout_rate=0.6)
        self.block3 = self.make_block(64, 128, num_convs=4, stride_down=2, dropout_rate=0.6)
        self.block4 = self.make_block(128, 256, num_convs=4, stride_down=2, dropout_rate=0.3)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def make_block(self, in_channels, out_channels, num_convs, stride_down, dropout_rate):
        layers = []
        for i in range(num_convs - 1):
            layers.append(nn.Conv2d(in_channels, out_channels,
                        kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels 
        layers.append(nn.Conv2d(in_channels, out_channels,kernel_size=3, stride=stride_down, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        # Input x is 64×64×3
        x = self.initial(x)     # (B, 32, 64, 64)
        x = self.block1(x)      # (B, 32, 32, 32)
        x = self.block2(x)      # (B, 64, 16, 16)
        x = self.block3(x)      # (B, 128, 8, 8)
        x = self.block4(x)      # (B, 256, 4, 4)
        x = self.gap(x)         # (B, 256, 1, 1)
        x = torch.flatten(x, 1) # (B, 256)
        x = self.fc(x)          # (B, num_classes)
        return x
