import torch

from torch_geometric.nn import GATv2Conv, LayerNorm,TransformerConv,global_mean_pool,RGATConv
from torch.nn import Linear,Sequential,ELU,Dropout
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
        x_geom,x_visual,x_prior = data.x_geom,data.x_visual,data.x_prior
        edge_index,batch = data.edge_index,data.batch
        edge_attr = data.edge_attr

        # 特征编码+融合
        h_visual = self.visual_encoder(x_visual)
        h_geom = self.geom_encoder(x_geom)
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
    def __init__(self, n_classes,num_relations=3):
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
        device = data.x_visual.device
        # 2. [关键步骤] 合并三种边并生成 edge_type
        # 我们需要在 forward 里把数据里分离的边合并成 RGAT 需要的格式
        edge_index_overlap = data.edge_index_overlap.to(device)
        edge_index_arch = data.edge_index_arch.to(device)
        edge_index_spatial = data.edge_index_spatial.to(device)

        # 定义关系 ID (0, 1, 2)
        # 0: Overlap (抑制)
        # 1: Arch (横向/牙弓)
        # 2: Spatial (纵向/空间)

        # 生成 edge_type 向量
        type_overlap = torch.zeros(edge_index_overlap.size(1), dtype=torch.long, device=device)
        type_arch = torch.ones(edge_index_arch.size(1), dtype=torch.long,device=device)
        type_spatial = torch.full((edge_index_spatial.size(1),), 2, dtype=torch.long,device=device)

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
    def __init__(self, n_classes, num_relations=4, num_iterations=3):
        """
        RecurrentAnatomyGAT: 迭代式解剖学感知图神经网络
        """
        super(RecurrentAnatomyGAT, self).__init__()

        self.n_classes = n_classes
        self.num_iterations = num_iterations

        # --- 1. 编码器 ---
        self.visual_encoder = torch.nn.Sequential(
            torch.nn.Linear(1024, 128), torch.nn.ReLU(), LayerNorm(128)
        )
        self.geom_encoder = torch.nn.Sequential(
            torch.nn.Linear(22, 128), torch.nn.ReLU(), LayerNorm(128)
        )
        self.prior_encoder = torch.nn.Sequential(
            torch.nn.Linear((n_classes + 1) * 2, 128), torch.nn.ReLU(), LayerNorm(128)
        )

        self.fused_dim = 128 * 3  # 384

        # --- 2. 关系 GAT 核心 ---
        self.rgat_layer_1 = RGATConv(
            in_channels=self.fused_dim,
            out_channels=48,
            num_relations=num_relations,
            heads=8, concat=True, dropout=0.2
        )
        self.norm_1 = LayerNorm(self.fused_dim)  # 48*8 = 384

        self.rgat_layer_2 = RGATConv(
            in_channels=self.fused_dim,
            out_channels=48,
            num_relations=num_relations,
            heads=8, concat=True, dropout=0.2
        )
        self.norm_2 = LayerNorm(self.fused_dim)

        # --- 3. 分类器 ---
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.fused_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, n_classes)
        )

    def forward(self, data, return_att=False):
        """
        参数:
            data: 图数据对象
            return_att (bool): 是否返回注意力权重 (仅在可视化时设为 True)
        """
        device = data.x_visual.device

        # 1. 准备边信息
        edge_index_overlap = data.edge_index_overlap.to(device)
        edge_index_arch = data.edge_index_arch.to(device)
        edge_index_spatial = data.edge_index_spatial.to(device)
        edge_index_vertical = data.edge_index_vertical.to(device)

        type_overlap = torch.zeros(edge_index_overlap.size(1), dtype=torch.long, device=device)
        type_arch = torch.ones(edge_index_arch.size(1), dtype=torch.long, device=device)
        type_vertical = torch.full((edge_index_vertical.size(1),), 2, dtype=torch.long, device=device)
        type_spatial = torch.full((edge_index_spatial.size(1),), 3, dtype=torch.long, device=device)

        # 拼接所有边
        edge_index = torch.cat([edge_index_overlap, edge_index_arch, edge_index_vertical, edge_index_spatial], dim=1)
        edge_type = torch.cat([type_overlap, type_arch, type_vertical, type_spatial], dim=0)

        # 2. 编码静态特征 (保留你的消融实验设置)
        h_visual = self.visual_encoder(data.x_visual)
        h_geom = self.geom_encoder(data.x_geom)

        # --- 你的消融设置 (根据需要保留或注释) ---
        # h_geom = torch.zeros_like(h_geom)
        h_visual = torch.zeros_like(h_visual)
        # -------------------------------------

        original_prior = data.x_prior.to(device)
        all_step_logits = []

        # 用于存储最后一次迭代的注意力权重
        final_attention_weights = None

        # 3. 初始化动态信念
        current_dynamic_belief = original_prior.clone()


        # --- 4. 迭代循环 (Recurrent Loop) ---
        for t in range(self.num_iterations):
            # A. 编码当前信念
            combined_prior_input = torch.cat([original_prior,current_dynamic_belief],dim=1)
            h_prior = self.prior_encoder(combined_prior_input)

            # B. 融合
            h_0 = torch.cat([h_visual, h_geom, h_prior], dim=-1)  # [N, 384]

            # C. RGAT 消息传递 (权重共享)
            h_1 = self.rgat_layer_1(h_0, edge_index, edge_type)
            h_1 = F.elu(h_1) + h_0
            h_1 = self.norm_1(h_1, data.batch)

            # D. RGAT Layer 2 (我们要提取这里的权重！)
            # 判断是否是最后一次迭代 且 需要返回权重
            is_last_iter = (t == self.num_iterations - 1)

            if return_att and is_last_iter:
                # 开启 return_attention_weights=True
                h_2, (att_edge_index, alpha) = self.rgat_layer_2(
                    h_1, edge_index, edge_type, return_attention_weights=True
                )

                # alpha 的形状是 [Num_Edges, Num_Heads] (例如 [E, 8])
                # 我们对多头取平均，得到每条边的综合权重
                final_attention_weights = alpha.mean(dim=1)
            else:
                # 正常前向传播
                h_2 = self.rgat_layer_2(h_1, edge_index, edge_type)

            h_2 = F.elu(h_2) + h_1
            h_2 = self.norm_2(h_2, data.batch)

            # D. 计算当前步的输出
            logits = self.classifier(h_2)  # [N, 49]
            all_step_logits.append(logits)

            # E. 更新下一次迭代的信念 (如果不是最后一步)
            if t < self.num_iterations - 1:
                # Softmax 获取概率分布
                probs = F.softmax(logits, dim=1)  # [N, 49]
                # 获取最大置信度
                confidence, _ = torch.max(probs, dim=1, keepdim=True)  # [N, 1]
                # 拼接成新的 x_prior [N, 50]
                # 这就是"信念传播"：GNN现在的输出变成了下一轮的输入
                current_dynamic_belief = torch.cat([probs, confidence], dim=1)

        # 5. 返回结果
        if return_att:
            # 返回: (所有步的logits, 最后一步的边权重, 最后一步的边索引)
            return all_step_logits, final_attention_weights, edge_index, edge_type

        return all_step_logits


class RecurrentAnatomyGATNew(torch.nn.Module):
    def __init__(self, n_classes, num_relations=4, num_iterations=3,
                 use_visual=True, use_geom=True):  # 新增消融控制开关
        """
        RecurrentAnatomyGAT: 具备动态边缘特征感知的迭代式神经符号图网络
        """
        super(RecurrentAnatomyGATNew, self).__init__()

        self.n_classes = n_classes
        self.num_iterations = num_iterations

        # 消融实验开关，告别 Hardcode
        self.use_visual = use_visual
        self.use_geom = use_geom

        # --- 1. 节点特征编码器 ---
        self.visual_encoder = torch.nn.Sequential(
            torch.nn.Linear(1024, 128), torch.nn.ReLU(), LayerNorm(128)
        )
        self.geom_encoder = torch.nn.Sequential(
            torch.nn.Linear(22, 128), torch.nn.ReLU(), LayerNorm(128)
        )
        self.prior_encoder = torch.nn.Sequential(
            torch.nn.Linear((n_classes + 1) * 2, 128), torch.nn.ReLU(), LayerNorm(128)
        )

        self.fused_dim = 128 * 3  # 节点融合特征维度: 384

        # --- 2. [核心新增] 边缘特征编码器 (Edge Encoder) ---
        # 重叠边是 6维 (带IoU), 其他边是 5维。我们统一填充到 6维 后送入编码器
        self.edge_dim = 64
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(6, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, self.edge_dim),
            LayerNorm(self.edge_dim)
        )

        # --- 3. 关系 GAT 核心 (引入 edge_dim 赋能动态注意力) ---
        self.rgat_layer_1 = RGATConv(
            in_channels=self.fused_dim,
            out_channels=48,
            num_relations=num_relations,
            heads=8, concat=True, dropout=0.2,
            edge_dim=self.edge_dim  # <--- [关键补丁] 让注意力机制看到边特征
        )
        self.norm_1 = LayerNorm(self.fused_dim)  # 48*8 = 384

        self.rgat_layer_2 = RGATConv(
            in_channels=self.fused_dim,
            out_channels=48,
            num_relations=num_relations,
            heads=8, concat=True, dropout=0.2,
            edge_dim=self.edge_dim  # <--- [关键补丁]
        )
        self.norm_2 = LayerNorm(self.fused_dim)

        # --- 4. 分类器 ---
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.fused_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, n_classes)
        )

    def forward(self, data, return_att=False):
        device = data.x_visual.device

        # ==========================================
        # 1. 拼接边索引 (Edge Index) 与 边类型 (Edge Type)
        # ==========================================
        edge_idx_list = [
            data.edge_index_overlap, data.edge_index_arch,
            data.edge_index_vertical, data.edge_index_spatial
        ]
        edge_index = torch.cat([e.to(device) for e in edge_idx_list], dim=1)

        type_overlap = torch.zeros(data.edge_index_overlap.size(1), dtype=torch.long, device=device)
        type_arch = torch.ones(data.edge_index_arch.size(1), dtype=torch.long, device=device)
        type_vertical = torch.full((data.edge_index_vertical.size(1),), 2, dtype=torch.long, device=device)
        type_spatial = torch.full((data.edge_index_spatial.size(1),), 3, dtype=torch.long, device=device)
        edge_type = torch.cat([type_overlap, type_arch, type_vertical, type_spatial], dim=0)

        # ==========================================
        # 2. [核心新增] 提取并对齐边缘特征 (Edge Attributes)
        # ==========================================
        # overlap 自带 IoU (6维)，其他关系缺乏 IoU (5维)，用 0 补齐第 6 维
        attr_overlap = data.edge_attr_overlap.to(device)
        attr_arch = F.pad(data.edge_attr_arch.to(device), (0, 1), value=0.0)
        attr_vert = F.pad(data.edge_attr_vertical.to(device), (0, 1), value=0.0)
        attr_spat = F.pad(data.edge_attr_spatial.to(device), (0, 1), value=0.0)

        raw_edge_attr = torch.cat([attr_overlap, attr_arch, attr_vert, attr_spat], dim=0)

        # 将原始几何属性映射为高维隐藏特征，供注意力机制使用
        encoded_edge_attr = self.edge_encoder(raw_edge_attr)

        # ==========================================
        # 3. 编码静态节点特征 (加入消融控制)
        # ==========================================
        h_visual = self.visual_encoder(data.x_visual)
        if not self.use_visual:
            h_visual = torch.zeros_like(h_visual)

        h_geom = self.geom_encoder(data.x_geom)
        if not self.use_geom:
            h_geom = torch.zeros_like(h_geom)

        original_prior = data.x_prior.to(device)
        all_step_logits = []
        final_attention_weights = None
        current_dynamic_belief = original_prior.clone()

        # ==========================================
        # 4. 迭代循环 (Working Memory Reasoning)
        # ==========================================
        for t in range(self.num_iterations):
            # A. 编码并融合当前信念 (Working Memory)
            combined_prior_input = torch.cat([original_prior, current_dynamic_belief], dim=1)
            h_prior = self.prior_encoder(combined_prior_input)
            h_0 = torch.cat([h_visual, h_geom, h_prior], dim=-1)  # [N, 384]

            # B. 关系感知与边缘特征注入的消息传递
            # 现在，RGAT 能够结合节点的视觉特征和它们之间的相对几何关系来决定权重
            h_1 = self.rgat_layer_1(h_0, edge_index, edge_type, edge_attr=encoded_edge_attr)
            h_1 = F.elu(h_1) + h_0
            h_1 = self.norm_1(h_1, data.batch)

            is_last_iter = (t == self.num_iterations - 1)

            # C. 第二层消息传递 (提取最终 Attention Map 供论文可视化)
            if return_att and is_last_iter:
                h_2, (att_edge_index, alpha) = self.rgat_layer_2(
                    h_1, edge_index, edge_type, edge_attr=encoded_edge_attr, return_attention_weights=True
                )
                final_attention_weights = alpha.mean(dim=1)
            else:
                h_2 = self.rgat_layer_2(h_1, edge_index, edge_type, edge_attr=encoded_edge_attr)

            h_2 = F.elu(h_2) + h_1
            h_2 = self.norm_2(h_2, data.batch)

            # D. 输出解码
            logits = self.classifier(h_2)
            all_step_logits.append(logits)

            # E. 动态信念更新 (State Evolution)
            if t < self.num_iterations - 1:
                probs = F.softmax(logits, dim=1)
                confidence, _ = torch.max(probs, dim=1, keepdim=True)
                current_dynamic_belief = torch.cat([probs, confidence], dim=1)

        if return_att:
            return all_step_logits, final_attention_weights, edge_index, edge_type

        return all_step_logits


class RecurrentAnatomyGATNew_A(torch.nn.Module):
    def __init__(self, n_classes, num_relations=4, num_iterations=3,
                 use_visual=True, use_geom=True, use_prior=True,
                 use_edge_features=True, spatial_only=False):  # [核心新增] spatial_only 开关
        """
        RecurrentAnatomyGAT: 具备动态边缘特征感知的迭代式神经符号图网络
        """
        super(RecurrentAnatomyGATNew_A, self).__init__()

        self.n_classes = n_classes
        self.num_iterations = num_iterations

        # 消融实验开关，告别 Hardcode
        self.use_visual = use_visual
        self.use_geom = use_geom
        self.use_prior = use_prior
        self.use_edge_features = use_edge_features
        self.spatial_only = spatial_only  # 是否退化为 Basic GAT

        # --- 1. 节点特征编码器 ---
        self.visual_encoder = torch.nn.Sequential(
            torch.nn.Linear(1024, 128), torch.nn.ReLU(), LayerNorm(128)
        )
        self.geom_encoder = torch.nn.Sequential(
            torch.nn.Linear(22, 128), torch.nn.ReLU(), LayerNorm(128)
        )
        self.prior_encoder = torch.nn.Sequential(
            torch.nn.Linear((n_classes + 1) * 2, 128), torch.nn.ReLU(), LayerNorm(128)
        )

        self.fused_dim = 128 * 3  # 节点融合特征维度: 384

        # --- 2. 边缘特征编码器 (Edge Encoder) ---
        self.edge_dim = 64
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(6, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, self.edge_dim),
            LayerNorm(self.edge_dim)
        )

        # --- 3. 关系 GAT 核心 ---
        self.rgat_layer_1 = RGATConv(
            in_channels=self.fused_dim,
            out_channels=48,
            num_relations=num_relations,
            heads=8, concat=True, dropout=0.2,
            edge_dim=self.edge_dim
        )
        self.norm_1 = LayerNorm(self.fused_dim)  # 48*8 = 384

        self.rgat_layer_2 = RGATConv(
            in_channels=self.fused_dim,
            out_channels=48,
            num_relations=num_relations,
            heads=8, concat=True, dropout=0.2,
            edge_dim=self.edge_dim
        )
        self.norm_2 = LayerNorm(self.fused_dim)

        # --- 4. 分类器 ---
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.fused_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, n_classes)
        )

    def forward(self, data, return_att=False):
        device = data.x_visual.device

        # ==========================================
        # 1. 拼接边索引 (Edge Index) 与 边类型 (Edge Type)
        # ==========================================
        edge_idx_list = [
            data.edge_index_overlap, data.edge_index_arch,
            data.edge_index_vertical, data.edge_index_spatial
        ]
        edge_index = torch.cat([e.to(device) for e in edge_idx_list], dim=1)

        type_overlap = torch.zeros(data.edge_index_overlap.size(1), dtype=torch.long, device=device)
        type_arch = torch.ones(data.edge_index_arch.size(1), dtype=torch.long, device=device)
        type_vertical = torch.full((data.edge_index_vertical.size(1),), 2, dtype=torch.long, device=device)
        type_spatial = torch.full((data.edge_index_spatial.size(1),), 3, dtype=torch.long, device=device)
        edge_type = torch.cat([type_overlap, type_arch, type_vertical, type_spatial], dim=0)

        # ==========================================
        # 2. 提取原始边缘特征 (Edge Attributes)
        # ==========================================
        attr_overlap = data.edge_attr_overlap.to(device)
        attr_arch = F.pad(data.edge_attr_arch.to(device), (0, 1), value=0.0)
        attr_vert = F.pad(data.edge_attr_vertical.to(device), (0, 1), value=0.0)
        attr_spat = F.pad(data.edge_attr_spatial.to(device), (0, 1), value=0.0)

        raw_edge_attr = torch.cat([attr_overlap, attr_arch, attr_vert, attr_spat], dim=0)

        # ==========================================
        # [核心消融拦截区]：Basic GAT 与 Edge 消融控制
        # ==========================================
        if self.spatial_only:
            # A. Basic GAT 退化：只保留空间边 (type == 3)
            mask = (edge_type == 3)
            edge_index = edge_index[:, mask]
            raw_edge_attr = raw_edge_attr[mask]
            # 抹平 edge_type，所有边变为 type 0，让 RGAT 退化为普通的单关系 GAT
            edge_type = torch.zeros_like(edge_type[mask])

        # B. 边缘特征编码
        encoded_edge_attr = self.edge_encoder(raw_edge_attr)

        if not self.use_edge_features:
            # C. 边缘特征消融：用全 0 覆盖，模型只能看到连接结构，看不到物理属性
            encoded_edge_attr = torch.zeros_like(encoded_edge_attr)

        # ==========================================
        # 3. 编码静态节点特征 (加入消融控制)
        # ==========================================
        h_visual = self.visual_encoder(data.x_visual)
        if not self.use_visual:
            h_visual = torch.zeros_like(h_visual)

        h_geom = self.geom_encoder(data.x_geom)
        if not self.use_geom:
            h_geom = torch.zeros_like(h_geom)

        original_prior = data.x_prior.to(device)
        if not self.use_prior:
            original_prior = torch.zeros_like(original_prior)

        all_step_logits = []
        final_attention_weights = None
        current_dynamic_belief = original_prior.clone()

        # ==========================================
        # 4. 迭代循环 (Working Memory Reasoning)
        # ==========================================
        for t in range(self.num_iterations):
            # A. 编码并融合当前信念
            combined_prior_input = torch.cat([original_prior, current_dynamic_belief], dim=1)
            h_prior = self.prior_encoder(combined_prior_input)
            h_0 = torch.cat([h_visual, h_geom, h_prior], dim=-1)  # [N, 384]

            # B. 关系感知与边缘特征注入的消息传递
            h_1 = self.rgat_layer_1(h_0, edge_index, edge_type, edge_attr=encoded_edge_attr)
            h_1 = F.elu(h_1) + h_0
            h_1 = self.norm_1(h_1)

            is_last_iter = (t == self.num_iterations - 1)

            # C. 第二层消息传递 (提取最终 Attention Map)
            h_2, (att_edge_index, alpha) = self.rgat_layer_2(
                h_1, edge_index, edge_type, edge_attr=encoded_edge_attr, return_attention_weights=True
            )
            final_attention_weights = alpha.mean(dim=1)

            h_2 = F.elu(h_2) + h_1
            h_2 = self.norm_2(h_2)

            # D. 输出解码
            logits = self.classifier(h_2)
            all_step_logits.append(logits)

            # E. 动态信念更新
            if t < self.num_iterations - 1:
                probs = F.softmax(logits, dim=1)
                confidence, _ = torch.max(probs, dim=1, keepdim=True)
                current_dynamic_belief = torch.cat([probs, confidence], dim=1)

        return all_step_logits, final_attention_weights, edge_index, edge_type


class SlotQualityRecurrentAnatomyGAT(torch.nn.Module):
    def __init__(self, n_classes=49, num_relations=4, num_iterations=5,
                 use_visual=False, use_geom=True, use_prior=True,
                 use_edge_features=True, spatial_only=False,
                 geom_dim=29, edge_attr_dim=11):
        super().__init__()
        self.n_classes = n_classes
        self.background_idx = n_classes - 1
        self.num_iterations = num_iterations
        self.use_visual = use_visual
        self.use_geom = use_geom
        self.use_prior = use_prior
        self.use_edge_features = use_edge_features
        self.spatial_only = spatial_only

        self.visual_encoder = torch.nn.Sequential(
            torch.nn.Linear(1024, 128), torch.nn.ReLU(), LayerNorm(128)
        )
        self.geom_encoder = torch.nn.Sequential(
            torch.nn.Linear(geom_dim, 128), torch.nn.ReLU(), LayerNorm(128)
        )
        self.prior_encoder = torch.nn.Sequential(
            torch.nn.Linear((n_classes + 1) * 2, 128), torch.nn.ReLU(), LayerNorm(128)
        )
        self.fused_dim = 128 * 3

        self.edge_dim = 64
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(edge_attr_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, self.edge_dim),
            LayerNorm(self.edge_dim)
        )

        self.rgat_layer_1 = RGATConv(
            in_channels=self.fused_dim,
            out_channels=48,
            num_relations=num_relations,
            heads=8,
            concat=True,
            dropout=0.2,
            edge_dim=self.edge_dim
        )
        self.norm_1 = LayerNorm(self.fused_dim)
        self.rgat_layer_2 = RGATConv(
            in_channels=self.fused_dim,
            out_channels=48,
            num_relations=num_relations,
            heads=8,
            concat=True,
            dropout=0.2,
            edge_dim=self.edge_dim
        )
        self.norm_2 = LayerNorm(self.fused_dim)

        self.slot_head = torch.nn.Sequential(
            torch.nn.Linear(self.fused_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, self.background_idx)
        )
        self.quality_head = torch.nn.Sequential(
            torch.nn.Linear(self.fused_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 2)
        )

    def _align_edge_attr(self, edge_attr):
        expected_dim = self.edge_encoder[0].in_features
        current_dim = edge_attr.size(1)
        if current_dim == expected_dim:
            return edge_attr
        if current_dim < expected_dim:
            return F.pad(edge_attr, (0, expected_dim - current_dim), value=0.0)
        return edge_attr[:, :expected_dim]

    def _build_edges(self, data, device):
        edge_idx_list = [
            data.edge_index_overlap, data.edge_index_arch,
            data.edge_index_vertical, data.edge_index_spatial
        ]
        edge_index = torch.cat([e.to(device) for e in edge_idx_list], dim=1)
        type_overlap = torch.zeros(data.edge_index_overlap.size(1), dtype=torch.long, device=device)
        type_arch = torch.ones(data.edge_index_arch.size(1), dtype=torch.long, device=device)
        type_vertical = torch.full((data.edge_index_vertical.size(1),), 2, dtype=torch.long, device=device)
        type_spatial = torch.full((data.edge_index_spatial.size(1),), 3, dtype=torch.long, device=device)
        edge_type = torch.cat([type_overlap, type_arch, type_vertical, type_spatial], dim=0)

        raw_edge_attr = torch.cat([
            self._align_edge_attr(data.edge_attr_overlap.to(device)),
            self._align_edge_attr(data.edge_attr_arch.to(device)),
            self._align_edge_attr(data.edge_attr_vertical.to(device)),
            self._align_edge_attr(data.edge_attr_spatial.to(device)),
        ], dim=0)

        if self.spatial_only:
            mask = edge_type == 3
            edge_index = edge_index[:, mask]
            raw_edge_attr = raw_edge_attr[mask]
            edge_type = torch.zeros_like(edge_type[mask])

        encoded_edge_attr = self.edge_encoder(raw_edge_attr)
        if not self.use_edge_features:
            encoded_edge_attr = torch.zeros_like(encoded_edge_attr)
        return edge_index, edge_type, encoded_edge_attr

    def forward(self, data, return_att=False):
        device = data.x_visual.device
        edge_index, edge_type, encoded_edge_attr = self._build_edges(data, device)

        h_visual = self.visual_encoder(data.x_visual)
        if not self.use_visual:
            h_visual = torch.zeros_like(h_visual)

        h_geom = self.geom_encoder(data.x_geom)
        if not self.use_geom:
            h_geom = torch.zeros_like(h_geom)

        original_prior = data.x_prior.to(device)
        if not self.use_prior:
            original_prior = torch.zeros_like(original_prior)

        current_dynamic_belief = original_prior.clone()
        all_step_outputs = []
        final_attention_weights = None

        for t in range(self.num_iterations):
            combined_prior_input = torch.cat([original_prior, current_dynamic_belief], dim=1)
            h_prior = self.prior_encoder(combined_prior_input)
            h_0 = torch.cat([h_visual, h_geom, h_prior], dim=-1)

            h_1 = self.rgat_layer_1(h_0, edge_index, edge_type, edge_attr=encoded_edge_attr)
            h_1 = F.elu(h_1) + h_0
            h_1 = self.norm_1(h_1)

            is_last_iter = t == self.num_iterations - 1
            if return_att and is_last_iter:
                h_2, (_, alpha) = self.rgat_layer_2(
                    h_1, edge_index, edge_type, edge_attr=encoded_edge_attr,
                    return_attention_weights=True
                )
                final_attention_weights = alpha.mean(dim=1)
            else:
                h_2 = self.rgat_layer_2(h_1, edge_index, edge_type, edge_attr=encoded_edge_attr)

            h_2 = F.elu(h_2) + h_1
            h_2 = self.norm_2(h_2)

            slot_logits = self.slot_head(h_2)
            quality_logits = self.quality_head(h_2)
            all_step_outputs.append({
                "slot_logits": slot_logits,
                "quality_logits": quality_logits,
            })

            if t < self.num_iterations - 1:
                slot_probs = F.softmax(slot_logits, dim=1)
                quality_probs = F.softmax(quality_logits, dim=1)
                bg_prob = quality_probs[:, :1]
                confidence, _ = torch.max(slot_probs, dim=1, keepdim=True)
                current_dynamic_belief = torch.cat([slot_probs, bg_prob, confidence], dim=1)

        if return_att:
            return all_step_outputs, final_attention_weights, edge_index, edge_type
        return all_step_outputs
