import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)  # Batch Normalization after first fully connected layer
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)   # Batch Normalization after second fully connected layer
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)  # Dropout layer

    def forward(self, feature_pairs):
        x = F.relu(self.bn1(self.fc1(feature_pairs)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze(1)      

class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPBinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)  # Higher dropout rate to prevent overfitting

    def forward(self, feature_pairs):
        x = F.relu(self.bn1(self.fc1(feature_pairs)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = torch.sigmoid(self.fc5(x))
        return x.squeeze(1)
