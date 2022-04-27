import numpy as np
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from trajnetplusplustools import Reader
from preprocess import process_datasets, process_dataset

file_paths = ["./train/real_data/biwi_hotel.ndjson", "./train/real_data/cff_06.ndjson", "./train/real_data/cff_07.ndjson", "./train/real_data/cff_08.ndjson", "./train/real_data/cff_09.ndjson", "./train/real_data/cff_10.ndjson", "./train/real_data/cff_12.ndjson", "./train/real_data/cff_13.ndjson", "./train/real_data/cff_14.ndjson", "./train/real_data/cff_15.ndjson", "./train/real_data/cff_16.ndjson", "./train/real_data/cff_17.ndjson", "./train/real_data/cff_18.ndjson", "./train/real_data/crowds_students001.ndjson", "./train/real_data/crowds_students003.ndjson", "./train/real_data/crowds_zara01.ndjson", "./train/real_data/crowds_zara03.ndjson", "./train/real_data/lcas.ndjson", "./train/real_data/wildtrack.ndjson"]

x, y = process_datasets(file_paths[0:len(file_paths)-1])
x, y = np.array(x), np.array(y)

trainX = Variable(torch.from_numpy(x))
trainY = Variable(torch.from_numpy(y))

x, y = process_dataset(file_paths[len(file_paths)-1])
x, y = np.array(x), np.array(y)

testX = Variable(torch.from_numpy(x))
testY = Variable(torch.from_numpy(y))

seq_length = 8

class LSTM(nn.Module):
    
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

        # num_classes => output size

    def forward(self, x, h_in, c_in):
        # Change: take hidden and cell state values and output their changed states
        
        # Propagate input through LSTM
        ula, (h_out, c_out) = self.lstm(x, (h_in, c_in))
        h_out = h_out.view(-1, self.hidden_size)
        

        out = self.fc(h_out)

        return out, h_out, c_out

num_epochs = 500
learning_rate = 0.01

input_size = 2
hidden_size = 50
num_layers = 1

num_classes = 2 

lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
lstm.double()

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

# Train the model
outputs = []
for epoch in range(num_epochs):

    h_in = Variable(torch.zeros(
            num_layers, trainX.size(0), hidden_size, dtype=torch.float64))
        
    c_in = Variable(torch.zeros(
            num_layers, trainX.size(0), hidden_size, dtype=torch.float64))

    outputs, _, _ = lstm(trainX, h_in, c_in)
    optimizer.zero_grad()
    
    # obtain the loss function
    loss = criterion(outputs, trainY)
    
    loss.backward()
    
    optimizer.step()
    if epoch % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

future = 12

input = torch.zeros((1, 20, 2), dtype=torch.float64)
input[:, 0:8, :] = testX[1]

h_in = Variable(torch.zeros(
            num_layers, input.size(0), hidden_size, dtype=torch.float64))
        
c_in = Variable(torch.zeros(
            num_layers, input.size(0), hidden_size, dtype=torch.float64))


for i in range(future):
    with torch.no_grad():
        out, h_in, c_in = lstm(input[:, i:i+8, :], h_in, c_in)
        h_in = h_in[None, :]
        input[:, 8+i, :] = out

print(input)



# # # # Step 1: Predict (x, y) - Done
# # # # Step 2: Once model is trained (seeing 8 predicting one more), produce 12 more values (one at a time passing in updated output each step)


# # # # Train with fresh h_0 & c_0 values
# # # # Testing 
# # # # - start with fresh h, c values 
# # # # - pass in first 8 (x, y) pairs
# # # # - we get (x, y) & (h1, c1)
# # # # - iterate on above to produce 12 more values


# # Tasks:
# # Need to train more for multiple pedestrians and multiple datasets
# # Plot trajectories to see what the real issue is
# # Try different hyperparameters
# # Try training on all but one datasets and test on the last one
# # Start using HAL  