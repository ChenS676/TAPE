
import torch 
import torch.nn.functional as F

class MF(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout,  node_num=None, cat_node_feat_mf=False,  data_name=None):
        super(MF, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.data = data_name
        if num_layers == 0:
            out_mf = out_channels
            if self.data=='ogbl-citation2':
                out_mf = 96

            self.emb =  torch.nn.Embedding(node_num, out_mf)
        else:
            self.emb =  torch.nn.Embedding(node_num, in_channels)

        if cat_node_feat_mf:
            in_channels = in_channels*2
    

        if num_layers == 1:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))

            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.invest = 1
        self.num_layers = num_layers
        self.cat_node_feat_mf = cat_node_feat_mf

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
            
        if self.data == 'ogbl-citation2':
            print('!!!! citaion2 !!!!!')
            torch.nn.init.normal_(self.emb.weight, std = 0.2)

        else: 
            self.emb.reset_parameters()



    def forward(self, x=None, adj_t=None):
        if self.invest == 1:
            print('layers in mlp: ', len(self.lins))
            self.invest = 0
        if self.cat_node_feat_mf and x != None:
            # print('xxxxxxx')
            x = torch.cat((x, self.emb.weight), dim=-1)

        else:
            x =  self.emb.weight


        if self.num_layers == 0:
            return self.emb.weight

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-1](x)

        return x

