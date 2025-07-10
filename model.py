import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        self.use_shortcut = stride != 1 or in_channels != out_channels

        # Transform input data to match output
        if self.use_shortcut:
            # The output of Conv2d will be the input to BatchNorm2d
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), 
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, feature_map_dict=None, prefix=""):
        output = self.conv1(x)
        output = self.bn1(output)
        output = torch.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        shortcut = self.shortcut(x) if self.use_shortcut else x
        output_shortcut = output + shortcut
        if feature_map_dict is not None:
            feature_map_dict[f"{prefix}.conv"] = output_shortcut
        
        output = torch.relu(output_shortcut)
        if feature_map_dict is not None:
            feature_map_dict[f"{prefix}.relu"] = output

        return output
    

class AudioCNN(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = nn.ModuleList([ResidualBlock(64, 64) for _ in range(3)])
        self.layer2 = nn.ModuleList(
            [ResidualBlock(64 if i == 0 else 128, 128, stride=2 if i == 0 else 1) for i in range(4)]
        )
        self.layer3 = nn.ModuleList(
            [ResidualBlock(128 if i == 0 else 256, 256, stride=2 if i == 0 else 1) for i in range(6)]
        )
        self.layer4 = nn.ModuleList(
            [ResidualBlock(256 if i == 0 else 512, 512, stride=2 if i == 0 else 1) for i in range(3)]
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # Flatten feature maps
        self.dropout = nn.Dropout(0.5) # Prevent overfitting during training
        self.linear_layer = nn.Linear(512, num_classes)

    def forward(self, x, return_feature_maps=False):
        if not return_feature_maps:
            x = self.conv1(x)

            # Pass x through every residual block for each layer
            for block in self.layer1:
                x = block(x)
            for block in self.layer2:
                x = block(x)
            for block in self.layer3:
                x = block(x)
            for block in self.layer4:
                x = block(x)
            
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1) # Reshape to (batch size, channels)
            x =self.dropout(x)
            x = self.linear_layer(x)
            return x
        else:
            feature_maps = {}
            x = self.conv1(x)
            feature_maps["conv1"] = x

            # Pass x through every residual block for each layer
            for index, block in enumerate(self.layer1):
                x = block(x, feature_maps, prefix=f"layer1.block{index}")
            feature_maps["layer1"] = x
            
            for index, block in enumerate(self.layer2):
                x = block(x, feature_maps, prefix=f"layer2.block{index}")
            feature_maps["layer2"] = x

            for index, block in enumerate(self.layer3):
                x = block(x, feature_maps, prefix=f"layer3.block{index}")
            feature_maps["layer3"] = x

            for index, block in enumerate(self.layer4):
                x = block(x, feature_maps, prefix=f"layer4.block{index}")
            feature_maps["layer4"] = x
            
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1) # Reshape to (batch size, channels)
            x =self.dropout(x)
            x = self.linear_layer(x)
            return x, feature_maps