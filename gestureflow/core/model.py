"""SkeletonCNN: classifies the 128x128 skeleton drawn by preprocessing.render."""

import torch.nn as nn


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SkeletonCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x

    def forward_with_embedding(self, x):
        """(logits, 128-d pooled features). The same computation as forward(), with the
        features the final layer sees also returned -- calibration compares these."""
        emb = self.head[:3](self.features(x))      # pool -> flatten -> dropout (off in eval)
        return self.head[3](emb), emb
