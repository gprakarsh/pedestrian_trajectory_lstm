import torch
from torch.autograd import Variable
from preprocess import process_dataset, process_datasets
import numpy as np
from lstm import LSTM



def main(file_paths, input_length, num_epochs, learning_rate, input_size, hidden_size, output_size, num_layers, save_path):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    print("Data processing started: ")
    x, y = process_datasets(file_paths)
    x, y = np.array(x), np.array(y)
    print("Datasets ready!")

    trainX = Variable(torch.from_numpy(x))
    trainY = Variable(torch.from_numpy(y))

    print("Initializing LSTM model")
    model = LSTM(output_size, input_size, hidden_size, num_layers, input_length).to(device)
    model.double()
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("LSTM model Initialized")

    print("Model training started")
    # Train the model
    outputs = []
    for epoch in range(num_epochs):
        loss = None
        for i in range(len(trainX)):
            currX = trainX[i].reshape(1,8,2).to(device)
            currY = trainY[i].reshape(1,2).to(device)

            h_in = Variable(torch.zeros(
                num_layers, currX.size(0), hidden_size, dtype=torch.float64, device=device))
            
            c_in = Variable(torch.zeros(
                num_layers, currX.size(0), hidden_size, dtype=torch.float64,device=device))
            

            outputs, _, _ = model(currX, h_in, c_in)
            # print(i, outputs, epoch)
            optimizer.zero_grad()
            
            loss = criterion(outputs, currY)
            # print(i, loss, epoch)

            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
        
    # torch.save({
    #     'state_dict': model.state_dict()
    #     # ,'optimizer_state_dict': optimizer.state_dict() # only if breaking batches
    #     }, save_path)
    torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    main(["./train/real_data/biwi_hotel.ndjson"], 8, 500, 0.01, 2, 50, 2, 1, "./lstm_1")