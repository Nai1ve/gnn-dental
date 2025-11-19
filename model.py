import torch

from torch_geometric.nn import GATv2Conv, LayerNorm,TransformerConv,global_mean_pool,RGATConv
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

    def __init__(self,in_channels=1030,hidden_channels=128,out_channels=49,heads=4,dropout_rate=0.2):
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




    def forward(self,data:Data):
        x,edge_index = data.x,data.edge_index

        x = self.conv1(x,edge_index)
        x = F.elu(x)
        x = F.dropout(x,p=self.dropout_rate,training=self.training)

        x = self.conv2(x,edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv3(x,edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        out = self.classifier(x)

        return out


class GAT_Numbering_Corrector_V3(torch.nn.Module):
    """
     基于GAT的编号纠正网络v0.2
     - 增加网络宽度
     - 优化最后的分类层
    """

    def __init__(self,in_channels=1030,hidden_channels=256,out_channels=49,heads=4,dropout_rate=0.2):
        """

        :param in_channels:
        :param hidden_channels:
        :param out_channels:
        :param heads:
        """
        super().__init__()
        self.dropout_rate = dropout_rate

        # 1. 扩展 1030-> 256 * heads
        self.conv1 = GATv2Conv(in_channels,hidden_channels,heads=heads)
        self.norm1 = LayerNorm(hidden_channels * heads)

        #2. 256 * heads ->256 * heads
        self.conv2 = GATv2Conv(hidden_channels* heads,hidden_channels,heads=heads)
        self.norm2 = LayerNorm(hidden_channels * heads)

        # #3. 【新增】第三层GAT，增加网络深度（可做实验验证是否有效）
        # self.conv3 = GATv2Conv(hidden_channels * heads,hidden_channels,heads=heads)


        #4. 【优化】改用MLP进行分类，不直接从256*4 -> 49
        self.classifier = Sequential(
            Linear(hidden_channels * heads,hidden_channels),
            ELU(),
            Dropout(p=self.dropout_rate),
            Linear(hidden_channels,out_channels)
        )



    def forward(self,data:Data):
        x,edge_index = data.x,data.edge_index

        x = self.conv1(x,edge_index)
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x,p=self.dropout_rate,training=self.training)

        x = self.conv2(x,edge_index)
        x = self.norm2(x)

        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        out = self.classifier(x)

        return out


class BaselineGNN(torch.nn.Module):
    def __init__(self,n_classes):
        super(BaselineGNN,self).__init__()

        #  多模态输入
        self.visual_encoder = torch.nn.Sequential(
            torch.nn.Linear(1024,128),
            torch.nn.ReLU(),
            LayerNorm(128)
        )

        self.geom_encoder = torch.nn.Sequential(
            torch.nn.Linear(6,128),
            torch.nn.ReLU(),
            LayerNorm(128)
        )

        self.prior_encoder = torch.nn.Sequential(
            torch.nn.Linear(50,128),
            torch.nn.ReLU(),
            LayerNorm(128)
        )
        self.fused_node_dim = 384 # 128 * 3


        #   边特征处理
        self.edge_feature_dim = 3 # [长度，角度x，角度y]
        self.encoded_edge_dim = 128

        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.edge_feature_dim,self.encoded_edge_dim),
            torch.nn.ReLU(),
            LayerNorm(self.encoded_edge_dim)
        )


        # GNN处理
        self.gnn_layer_1 = TransformerConv(
            self.fused_node_dim,48,heads=8,concat=True,
            edge_dim=self.encoded_edge_dim
        )
        self.gnn_norm_1 = LayerNorm(self.fused_node_dim)

        self.gnn_layer_2 = TransformerConv(
            self.fused_node_dim,48,heads=8,concat=True,
            edge_dim=self.encoded_edge_dim
        )
        self.gnn_norm_2 = LayerNorm(self.fused_node_dim)

        # 分类器头部
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.fused_node_dim,128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128,49)
        )


    def forward(self,data):
        # 数据
        x_graph,x_visual,x_prior = data.x_graph,data.x_visual,data.x_prior
        edge_index,batch = data.edge_index,data.batch
        edge_attr = data.edge_attr

        # 特征编码+融合
        h_visual = self.visual_encoder(x_visual)
        h_geom = self.geom_encoder(x_graph)
        h_prior = self.prior_encoder(x_prior)
        h_0 = torch.cat([h_visual,h_geom,h_prior],dim=-1)

        #边
        encoded_edge_attr = self.edge_encoder(edge_attr)

        h_1 = h_0 + F.elu(self.gnn_layer_1(h_0,edge_index,encoded_edge_attr))
        h_1 = self.gnn_norm_1(h_1,batch)

        h_2 = h_1 + F.elu(self.gnn_layer_2(h_1,edge_index,encoded_edge_attr))
        h_2 = self.gnn_norm_2(h_2,batch)

        logits = self.classifier(h_2)

        return logits

class AnatomyGAT(torch.nn.Module):
    def __init__(self, n_classes, num_relations=3):
        """
        AnatomyGAT: 解剖学感知的图神经网络

        参数:
            n_classes: 类别数 (49)
            num_relations: 边的类型数量 (这里是 3: Overlap, Arch, Spatial)
        """
        super(AnatomyGAT, self).__init__()

        # --- 1. 多模态输入编码器  ---

        self.visual_encoder = torch.nn.Sequential(
            torch.nn.Linear(1024, 128), torch.nn.ReLU(), LayerNorm(128)
        )
        self.geom_encoder = torch.nn.Sequential(
            torch.nn.Linear(6, 128), torch.nn.ReLU(), LayerNorm(128)
        )
        # 先验特征维度是 50 (49类 + 1置信度)
        self.prior_encoder = torch.nn.Sequential(
            torch.nn.Linear(n_classes + 1, 128), torch.nn.ReLU(), LayerNorm(128)
        )

        # 融合后的节点特征维度
        self.fused_dim = 128 * 3  # 384

        # --- 2. 关系 GAT 核心 (RGAT) ---
        # [核心创新点] 使用 RGATConv 替代 TransformerConv/GATv2
        # 它可以处理 edge_type

        self.rgat_layer_1 = RGATConv(
            in_channels=self.fused_dim,
            out_channels=48,
            num_relations=num_relations,  # 告诉它有3种边
            heads=8,
            concat=True,
            # 这里的 dropout 是注意力系数的 dropout
            dropout=0.2
        )
        self.norm_1 = LayerNorm(self.fused_dim)

        self.rgat_layer_2 = RGATConv(
            in_channels=self.fused_dim,
            out_channels=48,
            num_relations=num_relations,
            heads=8,
            concat=True,
            dropout=0.2
        )
        self.norm_2 = LayerNorm(self.fused_dim)

        # --- 3. 分类器头部 (与 Baseline 相同) ---
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.fused_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, n_classes)
        )

    def forward(self, data):
        # 1. 解包数据
        x_visual, x_geom, x_prior = data.x_visual, data.x_geom, data.x_prior

        # 2. [关键步骤] 合并三种边并生成 edge_type
        # 我们需要在 forward 里把数据里分离的边合并成 RGAT 需要的格式
        edge_index_overlap = data.edge_index_overlap
        edge_index_arch = data.edge_index_arch
        edge_index_spatial = data.edge_index_spatial

        # 定义关系 ID (0, 1, 2)
        # 0: Overlap (抑制)
        # 1: Arch (横向/牙弓)
        # 2: Spatial (纵向/空间)

        # 生成 edge_type 向量
        type_overlap = torch.zeros(edge_index_overlap.size(1), dtype=torch.long, device=data.x_visual.device)
        type_arch = torch.ones(edge_index_arch.size(1), dtype=torch.long, device=data.x_visual.device)
        type_spatial = torch.full((edge_index_spatial.size(1),), 2, dtype=torch.long, device=data.x_visual.device)

        # 拼接 edge_index 和 edge_type
        edge_index = torch.cat([edge_index_overlap, edge_index_arch, edge_index_spatial], dim=1)
        edge_type = torch.cat([type_overlap, type_arch, type_spatial], dim=0)

        # 3. 编码节点特征
        h_visual = self.visual_encoder(x_visual)
        h_geom = self.geom_encoder(x_geom)
        h_prior = self.prior_encoder(x_prior)
        h_0 = torch.cat([h_visual, h_geom, h_prior], dim=-1)  # [N, 384]

        # 4. RGAT 消息传递
        # Layer 1
        # 注意：RGATConv 需要传入 edge_type
        h_1 = self.rgat_layer_1(h_0, edge_index, edge_type)
        h_1 = F.elu(h_1)
        h_1 = h_0 + h_1  # 残差连接
        h_1 = self.norm_1(h_1, data.batch)

        # Layer 2
        h_2 = self.rgat_layer_2(h_1, edge_index, edge_type)
        h_2 = F.elu(h_2)
        h_2 = h_1 + h_2  # 残差连接
        h_2 = self.norm_2(h_2, data.batch)

        # 5. 分类
        logits = self.classifier(h_2)

        return logits


class RecurrentAnatomyGAT(torch.nn.Module):
    def __init__(self, n_classes, num_relations=3, num_iterations=3):
        """
        RecurrentAnatomyGAT: 迭代式解剖学感知图神经网络

        参数:
            num_iterations: 迭代推理的次数 (建议 2 或 3)
        """
        super(RecurrentAnatomyGAT, self).__init__()

        self.n_classes = n_classes
        self.num_iterations = num_iterations

        # --- 1. 编码器 (权重共享) ---
        self.visual_encoder = torch.nn.Sequential(
            torch.nn.Linear(1024, 128), torch.nn.ReLU(), LayerNorm(128)
        )
        self.geom_encoder = torch.nn.Sequential(
            torch.nn.Linear(6, 128), torch.nn.ReLU(), LayerNorm(128)
        )
        # 先验编码器：处理 [N, 50] 的向量
        self.prior_encoder = torch.nn.Sequential(
            torch.nn.Linear(n_classes + 1, 128), torch.nn.ReLU(), LayerNorm(128)
        )

        self.fused_dim = 128 * 3  # 384

        # --- 2. 关系 GAT 核心 (权重共享) ---
        # 在所有迭代步中，我们使用同一套 GNN 权重
        # 这迫使模型学习通用的"修正逻辑"，而不是死记硬背

        self.rgat_layer_1 = RGATConv(
            in_channels=self.fused_dim,
            out_channels=48,
            num_relations=num_relations,
            heads=8, concat=True, dropout=0.2
        )
        self.norm_1 = LayerNorm(self.fused_dim)

        self.rgat_layer_2 = RGATConv(
            in_channels=self.fused_dim,
            out_channels=48,
            num_relations=num_relations,
            heads=8, concat=True, dropout=0.2
        )
        self.norm_2 = LayerNorm(self.fused_dim)

        # --- 3. 分类器 (权重共享) ---
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.fused_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, n_classes)
        )

    def forward(self, data):
        device = data.x_visual.device

        # 1. 准备边信息 (只做一次)
        edge_index_overlap = data.edge_index_overlap.to(device)
        edge_index_arch = data.edge_index_arch.to(device)
        edge_index_spatial = data.edge_index_spatial.to(device)

        type_overlap = torch.zeros(edge_index_overlap.size(1), dtype=torch.long, device=device)
        type_arch = torch.ones(edge_index_arch.size(1), dtype=torch.long, device=device)
        type_spatial = torch.full((edge_index_spatial.size(1),), 2, dtype=torch.long, device=device)

        edge_index = torch.cat([edge_index_overlap, edge_index_arch, edge_index_spatial], dim=1)
        edge_type = torch.cat([type_overlap, type_arch, type_spatial], dim=0)

        # 2. 编码静态特征 (只做一次)
        h_visual = self.visual_encoder(data.x_visual)
        h_geom = self.geom_encoder(data.x_geom)

        # 3. 初始化动态信念 (Time step 0)
        # 初始信念来自检测器的原始输出
        current_prior_input = data.x_prior

        final_logits = None

        # --- 4. 迭代循环 (Recurrent Loop) ---
        for t in range(self.num_iterations):
            # A. 编码当前信念
            h_prior = self.prior_encoder(current_prior_input)

            # B. 融合
            h_0 = torch.cat([h_visual, h_geom, h_prior], dim=-1)  # [N, 384]

            # C. RGAT 消息传递 (权重共享)
            h_1 = self.rgat_layer_1(h_0, edge_index, edge_type)
            h_1 = F.elu(h_1) + h_0
            h_1 = self.norm_1(h_1, data.batch)

            h_2 = self.rgat_layer_2(h_1, edge_index, edge_type)
            h_2 = F.elu(h_2) + h_1
            h_2 = self.norm_2(h_2, data.batch)

            # D. 计算当前步的输出
            logits = self.classifier(h_2)  # [N, 49]
            final_logits = logits

            # E. 更新下一次迭代的信念 (如果不是最后一步)
            if t < self.num_iterations - 1:
                # Softmax 获取概率分布
                probs = F.softmax(logits, dim=1)  # [N, 49]
                # 获取最大置信度
                confidence, _ = torch.max(probs, dim=1, keepdim=True)  # [N, 1]
                # 拼接成新的 x_prior [N, 50]
                # 这就是"信念传播"：GNN现在的输出变成了下一轮的输入
                current_prior_input = torch.cat([probs, confidence], dim=1)

        return final_logits