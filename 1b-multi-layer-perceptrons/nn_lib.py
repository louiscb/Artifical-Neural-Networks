import torch
from torch.utils.data import dataset


class MLP(torch.nn.Module):
    def __init__(self, hidden_layers_size):
        super(MLP, self).__init__()
        self.hidden_layers = [torch.nn.Linear(5, hidden_layers_size[0])]
        for i in range(1, len(hidden_layers_size)):
            self.hidden_layers.append(torch.nn.Linear(hidden_layers_size[i - 1], hidden_layers_size[i]))
        self.output_layer = torch.nn.Linear(hidden_layers_size[-1], 1)
        self.activation_function = torch.nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation_function(x)
        return self.output_layer(x)


class MackeyGlassDataset(dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return torch.tensor(data), torch.tensor(label)

    def __len__(self):
        return self.data.shape[0]
