import torch.nn as nn

class LSTM(nn.Module):
    
    def __init__(self, output_size, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()

        self.output_size = output_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x, h_in, c_in):
        # Change: take hidden and cell state values and output their changed states
        
        # Propagate input through LSTM
        ula, (h_out, c_out) = self.lstm(x, (h_in, c_in))
        h_out = h_out.view(-1, self.hidden_size)
        

        out = self.fc(h_out)

        return out, h_out, c_out
