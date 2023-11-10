import timm
from torchinfo import summary
import torch.nn as nn
import torch

model_name = 'efficientnet_b0'

class EfficientNet(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained = True)
        num_filters = self.model.classifier.in_features
        layers = list(self.model.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        
        num_target_classes = 3
        self.classifier = nn.Linear(num_filters, num_target_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x)
        x = self.classifier(representations)
        x = self.softmax(x)
        return x
    
if __name__ == '__main__':
    model_name = 'efficientnet_b0'
    model = EfficientNet(model_name)
    summary(model, input_size = (8, 3, 256, 256), col_names = ["input_size", "output_size", "num_params"])