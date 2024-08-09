import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_shape, hidden_dim = 64, n_actions = 1):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim

        # print('input_shape: ', input_shape)
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        #GRU as LSTM
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self):
        return self.fc1.weight.new_zeros(1, self.hidden_dim)
    
    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        # -1 allows for dimension to be inferred for GRU cell compatibility
        h_reshape = hidden_state.reshape(-1, self.hidden_dim)
        h_state = self.rnn(x, h_reshape)
        q_val = self.fc2(h_state)
        return q_val, h_state
    
    #
    def copy_params(self, agent):
        for param, target_param in zip(agent.parameters(), self.parameters()):
            target_param.data.copy_(param.data)