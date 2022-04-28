import torch
from torch.autograd import Variable
from lstm import LSTM
from preprocess import process_datasets
import numpy as np


def main(file_paths, model_path, input_length, num_predictions, num_layers, input_size, hidden_size, output_size):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    print("Data processing started:")
    x, y = process_datasets(file_paths)
    x, y = np.array(x), np.array(y)

    testX = Variable(torch.from_numpy(x))
    testY = Variable(torch.from_numpy(y))
    print("Datasets ready!")

    print("Initializing model")
    input = torch.zeros((1, input_length + num_predictions, 2), dtype=torch.float64).to(device)
    input[:, 0:input_length, :] = testX[1]

    h_in = Variable(torch.zeros(
                num_layers, input.size(0), hidden_size, dtype=torch.float64)).to(device)
            
    c_in = Variable(torch.zeros(
                num_layers, input.size(0), hidden_size, dtype=torch.float64)).to(device)

    model = LSTM(output_size, input_size, hidden_size, num_layers, input_length).to(device)
    model.double()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("LSTM model initialized")

    print("Generating predictions")
    with torch.no_grad():
        for i in range(num_predictions):
            out, h_in, c_in = model(input[:, i:i+input_length, :], h_in, c_in)
            h_in = h_in[None, :]
            input[:, i + input_length, :] = out
    print("Generated predictions")
    print(input)

if __name__ == "__main__":
    main(["./train/real_data/wildtrack.ndjson"], "./lstm_1", 8, 12, 1, 2, 50, 2)
