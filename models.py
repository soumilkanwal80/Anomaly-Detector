import torch
import torchvision.models as tv
import torch.nn as nn
import torch.nn.functional as F

class SERLOC_A_Binary(nn.Module):
    def __init__(self, res101, res3d):
        super(SERLOC_A_Binary, self).__init__()
        self.res101 = res101
        self.res3d = res3d.train()
        
        self.hidden_dim = 512
        self.lstm1 = nn.LSTM(input_size=2048, hidden_size = self.hidden_dim, num_layers = 3, dropout = 0.2, bidirectional = True, batch_first = True)
        self.lstm2 = nn.LSTM(input_size=2048, hidden_size = self.hidden_dim, num_layers = 3, dropout = 0.2, bidirectional = True, batch_first = True)
        self.linear1 = nn.Linear(512*3, 512)
        self.final1 = nn.Linear(512, 1)
        #self.final4 = nn.Linear(512, 13)
        
    def forward(self, x, x3d, fv):
        y_1 = []
        y_2 = []
        y_3 = []
        #print(x.shape, x3d.shape, fv.shape)
        with torch.no_grad():
            self.h0_1 = torch.zeros(3*2, 1, self.hidden_dim).to(device)
            self.c0_1 = torch.zeros(3*2, 1, self.hidden_dim).to(device)
            self.h0_2 = torch.zeros(3*2, 1, self.hidden_dim).to(device)
            self.c0_2 = torch.zeros(3*2, 1, self.hidden_dim).to(device)
            for i in range(x.shape[0]):
                #print(x[i].shape)
                tmp = self.res101(x[i])
                tmp = tmp.mean([2, 3])
                tmp_fv = fv[i]
                #tmp = torch.cat([tmp, tmp_fv], 1)
                y_1.append(tmp)
                y_2.append(tmp_fv)
            
        y_4 = self.res3d(x3d)
        y_4 = y_4[:, :, -1, -1, -1]
        y_1 = torch.stack(y_1)
        y_2 = torch.stack(y_2)
        #print(y_1.shape, y_2.shape)
        x_1, _ = self.lstm1(y_1, (self.h0_1, self.c0_1))
        x_1 = x_1[:, -1, 0:512]

        x_2, _ = self.lstm2(y_2, (self.h0_2, self.c0_2))
        x_2 = x_2[:, -1, 0:512]
        
        x = torch.cat([x_1, x_2, y_4], axis = 1)
        x = self.final1(F.relu(self.linear1(x)))
        return x

class SERLOC_B_Anomaly(nn.Module):
    def __init__(self, res101, res3d):
        super(SERLOC_B_Anomaly, self).__init__()
        self.res101 = res101
        self.res3d = res3d.train()
        
        self.hidden_dim = 512
        #Linear layers
        self.linear_o1 = nn.Linear(2048, 512)
        self.linear_res1 = nn.Linear(2048, 512)
        self.final1=nn.Linear(512*3, 512)
        self.final2 = nn.Linear(512, 12)
        
        self.dropout1=nn.Dropout(p=0.4)

    def forward(self, x, x3d, fv):
        #res features
        with torch.no_grad():
            x_res = self.res101(x)
            x_res = x_res.mean([2, 3])
            
        x_res1=F.relu(self.linear_res1(x_res))
        #object features
        x_o1=F.relu(self.linear_o1(fv))
        #res3d features
        x_res3d1 = self.res3d(x3d)
        x_res3d1 = x_res3d1[:, :, -1, -1, -1]
        
        cat1 = torch.cat([x_res1, x_o1, x_res3d1], axis = 1)
        dense1=F.relu(self.final1(cat1))
        drop1=self.dropout1(dense1)
        x = self.final2(drop1)
        return x