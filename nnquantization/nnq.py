import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleNN()
print("Original weights of fc1 before quantization:")
print(model.fc1.weight.data)


weights = model.fc1.weight.data.cpu().numpy()

weights_reshaped = weights.reshape(-1, 1)


k = 3
kmeans = KMeans(n_clusters=k, random_state=0).fit(weights_reshaped)


centroids = kmeans.cluster_centers_.flatten()
quantized_weights_indices = kmeans.predict(weights_reshaped)
quantized_weights = np.array([centroids[i] for i in quantized_weights_indices])


quantized_weights = quantized_weights.reshape(weights.shape)


model.fc1.weight.data = torch.tensor(quantized_weights, dtype=torch.float32)

print("\nWeights of fc1 after quantization:")
print(model.fc1.weight.data)


input_data = torch.randn(1, 10)
output = model(input_data)
print("\nModel output with quantized weights:")
print(output)
