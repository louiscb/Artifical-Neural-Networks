from mackey_glass_ts_data_generation import *
from nn_lib import *
from torch.utils.data import DataLoader


def main():
    ts = create_time_series()
    #plot_time_series(ts)
    train, val, test = create_data_sets(ts, .7)
    train_set = MackeyGlassDataset(train[0], train[1])
    validation_set = MackeyGlassDataset(val[0], val[1])
    create_and_assess_model([9], 0.01, 0.9, 0.1, train_set, validation_set)
    test_set = MackeyGlassDataset(test[0], test[1])

def create_and_assess_model(hidden_layer_sizes, learning_rate, alpha, regularization, train_set, val_set):
    model = MLP(hidden_layer_sizes)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=alpha, weight_decay=regularization)
    train_loss, val_loss = train_model(model, optimizer, train_set, val_set)
    prefix = "lr={}_hidden_layers={}_lambda={}".format(learning_rate, hidden_layer_sizes, regularization)
    torch.save(model, prefix + '.model')
    write_results_to_file(prefix, train_loss, val_loss)


def write_results_to_file(prefix, train_loss, val_loss):
    f = open(prefix + '.txt', 'w')
    f.write(str(len(train_loss)) + '\n')
    f.write('TRAINING LOSS\n')
    for loss in train_loss:
        f.write(str(loss) + '\n')
    f.write('VALIDATION LOSS\n')
    for loss in val_loss:
        f.write(str(loss) + '\n')


def train_model(model, optimizer, train_set, val_set):
    loader = DataLoader(train_set, batch_size=train_set.__len__())
    criterion = torch.nn.MSELoss()
    train_loss = []
    val_loss = []
    curr_val_loss = 0
    prev_val_loss = float('inf')
    epochs = 1
    # train until validation loss stops improving "early stopping"
    while curr_val_loss < prev_val_loss:
        if epochs != 1:
            prev_val_loss = curr_val_loss
        curr_train_loss = execute_training_loop(model, optimizer, criterion, loader)
        train_loss.append(curr_train_loss)
        curr_val_loss = evaluate_model(model, val_set)
        val_loss.append(curr_val_loss)
        print(epochs, str(curr_train_loss), str(curr_val_loss))
        epochs += 1
    return train_loss, val_loss

def execute_training_loop(model, optimizer, criterion, loader):
    model.train()
    batch_loss = 0
    # train loop
    for x, y in loader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        # accounting
        batch_loss += loss.item()
    return batch_loss


def evaluate_model(model, data_set):
    loader = DataLoader(data_set, batch_size=data_set.__len__())
    criterion = torch.nn.MSELoss()
    model.eval()
    batch_loss = 0
    for x, y in loader:
        outputs = model(x)
        loss = criterion(outputs, y)
        batch_loss = loss.item()
    return batch_loss


main()
