from mackey_glass_ts_data_generation import *
from nn_lib import *
from torch.utils.data import DataLoader
import itertools
import copy


def main():
    ts = create_time_series()
    # plot_time_series(ts)
    train, val, test = create_data_sets(ts, .7)
    test_set = MackeyGlassDataset(test[0], test[1])
    model = torch.load('models/lr=0.01_hidden_layers=[3, 5]_lambda=0.0001.model')
    best_mse, best_predictions, targets = evaluate_model(model, test_set)
    model = torch.load('models/lr=0.01_hidden_layers=[4, 6]_lambda=0.0001.model')
    worst_mse, worst_predictions = evaluate_model(model, test_set)[:2]
    print(str(best_mse), str(worst_mse))
    #plot_predictions_and_targets(best_predictions, worst_predictions, targets)

    #train_set = MackeyGlassDataset(train[0], train[1])
    #validation_set = MackeyGlassDataset(val[0], val[1])
    #model = create_and_assess_model(train_set, validation_set, [6, 6], learning_rate=.01, alpha=.9, regularization=0.0001)


def create_and_assess_model(train_set, val_set, hidden_layer_sizes, learning_rate, alpha, regularization):
    ITERATIONS = 5
    train_loss = []
    val_loss = []
    model = None
    while ITERATIONS > 0:
        model = MLP(hidden_layer_sizes)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=alpha, weight_decay=regularization)
        model, train_loss_i, val_loss_i = train_model(model, optimizer, train_set, val_set)
        train_loss.append(train_loss_i)
        val_loss.append(val_loss_i)
        ITERATIONS -= 1
        print(ITERATIONS)
    train_loss = pad_numpify_and_average(train_loss)
    val_loss = pad_numpify_and_average(val_loss)
    prefix = "lr={}_hidden_layers={}_lambda={}".format(learning_rate, hidden_layer_sizes, regularization)
    torch.save(model, prefix + '.model')
    write_results_to_file(prefix, train_loss, val_loss)
    return model


def pad_numpify_and_average(list_of_lists):
    x = np.array(list(itertools.zip_longest(*list_of_lists, fillvalue=np.nan))).T
    return np.nanmean(x, axis=0)


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
    loader = DataLoader(train_set, shuffle=True)
    criterion = torch.nn.MSELoss()
    train_loss = []
    val_loss = []
    curr_val_loss = 0
    prev_val_loss = float('inf')
    epochs = 1
    prev_model = None
    # train until validation loss stops improving "early stopping"
    while curr_val_loss < prev_val_loss:
        if epochs != 1:
            prev_val_loss = curr_val_loss
            prev_model = copy.deepcopy(model)
        curr_train_loss = execute_training_loop(model, optimizer, criterion, loader)
        train_loss.append(curr_train_loss)
        curr_val_loss = evaluate_model(model, val_set)[0]
        val_loss.append(curr_val_loss)
        epochs += 1
    return prev_model, train_loss, val_loss


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
    predictions = None
    targets = None
    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            targets = y
            predictions = outputs
            loss = criterion(outputs, y)
            batch_loss = loss.item()
    return batch_loss, predictions.detach().numpy(), targets.detach().numpy()

def plot_predictions_and_targets(best, worst, targets):
    T = len(best)
    plt.plot(list(range(T)), best, color='blue')
    plt.plot(list(range(T)), worst, color='red')
    plt.plot(list(range(T)), targets, color='black')
    plt.show()


main()
