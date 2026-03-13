import torch
import torch.nn as nn

class Gaussian_Position(nn.Module):
    def __init__(self, var_dim_feature, var_dim_time, var_num_gaussian=10):
        super(Gaussian_Position, self).__init__()
        var_embedding = torch.zeros([var_num_gaussian, var_dim_feature], dtype=torch.float)
        self.var_embedding = nn.Parameter(var_embedding, requires_grad=True)
        nn.init.xavier_uniform_(self.var_embedding)
        var_position = torch.arange(0.0, var_dim_time).unsqueeze(1).repeat(1, var_num_gaussian)
        self.var_position = nn.Parameter(var_position, requires_grad=False)
        var_mu = torch.arange(0.0, var_dim_time, var_dim_time/var_num_gaussian).unsqueeze(0)
        self.var_mu = nn.Parameter(var_mu, requires_grad=True)
        var_sigma = torch.tensor([50.0] * var_num_gaussian).unsqueeze(0)
        self.var_sigma = nn.Parameter(var_sigma, requires_grad=True)

    def calculate_pdf(self, var_position, var_mu, var_sigma):
        var_pdf = var_position - var_mu
        var_pdf = - var_pdf * var_pdf
        var_pdf = var_pdf / var_sigma / var_sigma / 2
        var_pdf = var_pdf - torch.log(var_sigma)
        return var_pdf

    def forward(self, var_input):
        var_pdf = self.calculate_pdf(self.var_position, self.var_mu, self.var_sigma)
        var_pdf = torch.softmax(var_pdf, dim=-1)
        var_position_encoding = torch.matmul(var_pdf, self.var_embedding)
        return var_input + var_position_encoding.unsqueeze(0)

class THAT_EncoderLayer(nn.Module):
    def __init__(self, var_dim_feature, var_num_head=10, var_size_cnn=[1, 3, 5]):
        super(THAT_EncoderLayer, self).__init__()
        self.layer_norm_0 = nn.LayerNorm(var_dim_feature, eps=1e-6)
        self.layer_attention = nn.MultiheadAttention(var_dim_feature, var_num_head, batch_first=True)
        self.layer_dropout_0 = nn.Dropout(0.1)
        self.layer_norm_1 = nn.LayerNorm(var_dim_feature, 1e-6)
        layer_cnn = []
        for var_size in var_size_cnn:
            layer = nn.Sequential(nn.Conv1d(var_dim_feature, var_dim_feature, var_size, padding="same"),
                                  nn.BatchNorm1d(var_dim_feature),
                                  nn.Dropout(0.1),
                                  nn.LeakyReLU())
            layer_cnn.append(layer)
        self.layer_cnn = nn.ModuleList(layer_cnn)
        self.layer_dropout_1 = nn.Dropout(0.1)

    def forward(self, var_input):
        var_t = var_input
        var_t = self.layer_norm_0(var_t)
        var_t, _ = self.layer_attention(var_t, var_t, var_t)
        var_t = self.layer_dropout_0(var_t)
        var_t = var_t + var_input
        var_s = self.layer_norm_1(var_t)
        var_s = torch.permute(var_s, (0, 2, 1))
        var_c = torch.stack([layer(var_s) for layer in self.layer_cnn], dim=0)
        var_s = torch.sum(var_c, dim=0) / len(self.layer_cnn)
        var_s = self.layer_dropout_1(var_s)
        var_s = torch.permute(var_s, (0, 2, 1))
        return var_s + var_t

class THAT_Encoder(nn.Module):
    def __init__(self, projection_dim=512, time_steps=3000, features=270):
        super(THAT_Encoder, self).__init__()
        # ------------------- Left Branch -------------------
        self.layer_left_pooling = nn.AvgPool1d(kernel_size=20, stride=20)
        self.layer_left_gaussian = Gaussian_Position(features, time_steps // 20)
        self.layer_left_encoder = nn.ModuleList([THAT_EncoderLayer(features, 10, [1, 3, 5]) for _ in range(4)])
        self.layer_left_norm = nn.LayerNorm(features, eps=1e-6)
        self.layer_left_cnn_0 = nn.Conv1d(features, 128, kernel_size=8)
        self.layer_left_cnn_1 = nn.Conv1d(features, 128, kernel_size=16)
        self.layer_left_dropout = nn.Dropout(0.5)
        
        # ------------------- Right Branch -------------------
        self.layer_right_pooling = nn.AvgPool1d(kernel_size=20, stride=20)
        var_dim_right = time_steps // 20
        self.layer_right_encoder = nn.ModuleList([THAT_EncoderLayer(var_dim_right, 10, [1, 2, 3]) for _ in range(1)])
        self.layer_right_norm = nn.LayerNorm(var_dim_right, eps=1e-6)
        self.layer_right_cnn_0 = nn.Conv1d(var_dim_right, 16, kernel_size=2)
        self.layer_right_cnn_1 = nn.Conv1d(var_dim_right, 16, kernel_size=4)
        self.layer_right_dropout = nn.Dropout(0.5)
        
        self.layer_leakyrelu = nn.LeakyReLU()
        # 原特征融合维度 288 -> 投影到联合空间
        self.projection = nn.Linear(288, projection_dim)

    def forward(self, x):
        # 展平输入空间维度: [Batch, 3000, 3, 3, 30] -> [Batch, 3000, 270]
        if x.dim() == 5:
            x = x.view(x.size(0), x.size(1), -1)
            
        var_t = x
        
        # --- Left ---
        var_left = torch.permute(var_t, (0, 2, 1))
        var_left = self.layer_left_pooling(var_left)
        var_left = torch.permute(var_left, (0, 2, 1))
        var_left = self.layer_left_gaussian(var_left)
        for layer in self.layer_left_encoder: var_left = layer(var_left)
        var_left = self.layer_left_norm(var_left)
        var_left = torch.permute(var_left, (0, 2, 1))
        var_left_0 = self.layer_leakyrelu(self.layer_left_cnn_0(var_left))
        var_left_1 = self.layer_leakyrelu(self.layer_left_cnn_1(var_left))
        var_left = self.layer_left_dropout(torch.concat([torch.sum(var_left_0, dim=-1), torch.sum(var_left_1, dim=-1)], dim=-1))
        
        # --- Right ---
        var_right = torch.permute(var_t, (0, 2, 1))
        var_right = self.layer_right_pooling(var_right)
        for layer in self.layer_right_encoder: var_right = layer(var_right)
        var_right = self.layer_right_norm(var_right)
        var_right = torch.permute(var_right, (0, 2, 1))
        var_right_0 = self.layer_leakyrelu(self.layer_right_cnn_0(var_right))
        var_right_1 = self.layer_leakyrelu(self.layer_right_cnn_1(var_right))
        var_right = self.layer_right_dropout(torch.concat([torch.sum(var_right_0, dim=-1), torch.sum(var_right_1, dim=-1)], dim=-1))
        
        # --- Project ---
        projected = self.projection(torch.concat([var_left, var_right], dim=-1))
        return projected