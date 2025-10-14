import torch

from torch_geometric.nn import GATv2Conv
from torch.nn import Linear,Sequential,ReLU,ELU,Dropout
from torch_geometric.data import Data
import torch.nn.functional as F


class GAT_Numbering_Corrector(torch.nn.Module):
    """
     基于GAT的编号纠正网络v0.1
    """

    def __init__(self,in_channels=55,hidden_channels=128,out_channels=49,heads=4):
        """

        :param in_channels:
        :param hidden_channels:
        :param out_channels:
        :param heads:
        """
        super().__init__()

        # 1. 扩展 54-> 128 * heads
        self.conv1 = GATv2Conv(in_channels,hidden_channels,heads=heads)

        #2. 128 * heads ->128 * heads
        self.conv2 = GATv2Conv(hidden_channels* heads,hidden_channels,heads=heads)

        #3. 128 * heads -> 49
        self.classifier = Linear(hidden_channels *heads,out_channels)



    def forward(self,data:Data):
        x,edge_index = data.x,data.edge_index

        x = self.conv1(x,edge_index)
        x = F.elu(x)
        x = F.dropout(x,p=0.5,training=self.training)

        x = self.conv2(x,edge_index)
        x = F.elu(x)

        out = self.classifier(x)

        return out




class GAT_Numbering_Corrector_V2(torch.nn.Module):
    """
     基于GAT的编号纠正网络v0.2
     - 增加网络深度
     - 调整dropout参数
     - 优化最后的分类层
    """

    def __init__(self,in_channels=55,hidden_channels=128,out_channels=49,heads=4,dropout_rate=0.2):
        """

        :param in_channels:
        :param hidden_channels:
        :param out_channels:
        :param heads:
        """
        super().__init__()
        self.dropout_rate = dropout_rate

        # 1. 扩展 55-> 128 * heads
        self.conv1 = GATv2Conv(in_channels,hidden_channels,heads=heads)

        #2. 128 * heads ->128 * heads
        self.conv2 = GATv2Conv(hidden_channels* heads,hidden_channels,heads=heads)

        #3. 【新增】第三层GAT，增加网络深度（可做实验验证是否有效）
        self.conv3 = GATv2Conv(hidden_channels * heads,hidden_channels,heads=heads)


        #4. 【优化】改用MLP进行分类，不直接从128*4 -> 49
        self.classifier = Sequential(
            Linear(hidden_channels * heads,hidden_channels),
            ELU(),
            Dropout(p=self.dropout_rate),
            Linear(hidden_channels,out_channels)
        )

        #3. 128 * heads -> 49
        self.classifier = Linear(hidden_channels *heads,out_channels)



    def forward(self,data:Data):
        x,edge_index = data.x,data.edge_index

        x = self.conv1(x,edge_index)
        x = F.elu(x)
        x = F.dropout(x,p=self.dropout_rate,training=self.training)

        x = self.conv2(x,edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv2(x,edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        out = self.classifier(x)

        return out