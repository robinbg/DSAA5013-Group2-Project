import torch
import torch.nn as nn
import math
import timm

class EffNetV2(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True, pretrained_path=None):
        super(EffNetV2, self).__init__()
        self.num_classes = num_classes
        self.model = timm.create_model(pretrained_path, pretrained=pretrained, num_classes=num_classes)



    def forward(self, x, labels = None):
        x = self.model(x)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(x.view(-1, self.num_classes), labels.view(-1))
            return {'loss':loss, 'logits':x}
        return x
