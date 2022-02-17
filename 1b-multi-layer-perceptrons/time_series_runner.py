from mackey_glass_ts_data_generation import *
from nn_lib import *
import torch.utils.data.dataloader as DataLoader

def main():
    ts = create_time_series()
    train, val, test = create_data_sets(ts, .7)
    model = MLP([9])
    train_set = MackeyGlassDataset(train[0], train[1])
    validation_set = MackeyGlassDataset(val[0], val[1])
    test_set = MackeyGlassDataset(test[0], test[1])

def train_model(model, learning_rate, data_set):
    loader = DataLoader(data_set, batch_size=data_set.__len__())
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    model.train()
    for x, y in loader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

def evaluate_model(model, data_set):


main()