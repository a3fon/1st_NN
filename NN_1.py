import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

df = pd.read_csv('data.csv', sep=';')
df_features = df.iloc[:, :36]
df_labels = df.iloc[:, 36]
#print(df_features)

train_features, val_features, train_labels, val_labels = train_test_split(
    df_features, df_labels, test_size=0.2, random_state=0
)

#print(train_features.shape)
#print(val_labels.shape)

# Encode the labels with LabelEncoder
label_encoder = LabelEncoder()
encoded_train_labels = label_encoder.fit_transform(train_labels)
encoded_val_labels = label_encoder.transform(val_labels)

# Create the MinMaxScaler
scaler = MinMaxScaler()

# Normalize the training set
scaler.fit(train_features)
train_features_normalized = scaler.transform(train_features)

# Apply the same transformation to the validation set
val_features_normalized = scaler.transform(val_features)

# Convert to pandas DataFrame
train_features = pd.DataFrame(train_features_normalized, columns=train_features.columns)
val_features = pd.DataFrame(val_features_normalized, columns=val_features.columns)

# Converting data to PyTorch tensors
train_features_tensor = torch.tensor(train_features.values, dtype=torch.float32)
train_labels_tensor = torch.tensor(encoded_train_labels, dtype=torch.long)

# If the encoded_train_labels is a numpy array
#counts = np.bincount(encoded_train_labels)

val_features_tensor = torch.tensor(val_features.values, dtype=torch.float32)
val_labels_tensor = torch.tensor(encoded_val_labels, dtype=torch.long)

# If the encoded_train_labels is a numpy array
#counts = np.bincount(encoded_val_labels)

# Define the datasets
train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
val_dataset = TensorDataset(val_features_tensor, val_labels_tensor)

# Define the dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

class ReceptorNet(nn.Module):
    def __init__(self):
        super(ReceptorNet, self).__init__()
        self.fc_1 = nn.Linear(36, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.output = nn.Linear(256, 3)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x_out = self.output(x)  # No sigmoid is used here
        return x_out

# Create the model
model = ReceptorNet()

# Use CrossEntropyLoss for multiclass classification
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        # CrossEntropyLoss expects labels of type long
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Accuracy check
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on the validation set: {100 * correct / total:.2f}%')

