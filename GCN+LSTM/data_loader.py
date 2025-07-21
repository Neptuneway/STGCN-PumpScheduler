import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# All comments below are translated to English, and data-related parameters are to be passed from config.py
class WaterNetworkDataset(Dataset):
    def __init__(self, 
                 pressure_data, 
                 head_loss_data, 
                 seq_length=12, 
                 target_step=12,
                 scaler_pressure=None,
                 scaler_head_loss=None):
        """
        Water network dataset
        Args:
            pressure_data: Pressure data [num_samples, num_nodes]
            head_loss_data: Head loss data [num_samples, num_outputs]
            seq_length: Input sequence length
            target_step: Prediction target step
            scaler_pressure: Standard scaler for pressure data
            scaler_head_loss: Standard scaler for head loss data
        """
        self.pressure_data = pressure_data
        self.head_loss_data = head_loss_data
        self.seq_length = seq_length
        self.target_step = target_step
        
        # 数据标准化
        if scaler_pressure is None:
            self.scaler_pressure = StandardScaler()
            self.pressure_data = self.scaler_pressure.fit_transform(pressure_data)
        else:
            self.scaler_pressure = scaler_pressure
            self.pressure_data = self.scaler_pressure.transform(pressure_data)
            
        if scaler_head_loss is None:
            self.scaler_head_loss = StandardScaler()
            self.head_loss_data = self.scaler_head_loss.fit_transform(head_loss_data)
        else:
            self.scaler_head_loss = scaler_head_loss
            self.head_loss_data = self.scaler_head_loss.transform(head_loss_data)
        
        # 计算有效样本数量
        self.num_samples = len(pressure_data) - seq_length - target_step + 1
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 获取输入序列
        x = self.pressure_data[idx:idx + self.seq_length]  # [seq_length, num_nodes]
        x = x[..., None]  # 增加特征维度，变为[seq_length, num_nodes, 1]
        
        # 获取目标（未来第target_step步的水头损失）
        target_idx = idx + self.seq_length + self.target_step - 1
        y = self.head_loss_data[target_idx]  # [num_outputs]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)

def calc_gso(adj_matrix, gso_type='sym_renorm_adj'):
    """
    计算图移位算子 (Graph Shift Operator)
    Args:
        adj_matrix: 邻接矩阵 [n, n]
        gso_type: 归一化类型
    Returns:
        gso: 归一化后的邻接矩阵
    """
    n = adj_matrix.shape[0]
    
    if gso_type == 'sym_norm_lap':
        # 对称归一化拉普拉斯矩阵: L_sym = I - D^(-1/2) * A * D^(-1/2)
        degree = np.sum(adj_matrix, axis=1)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
        degree_inv_sqrt = np.diag(degree_inv_sqrt)
        gso = np.eye(n) - degree_inv_sqrt @ adj_matrix @ degree_inv_sqrt
        
    elif gso_type == 'rw_norm_lap':
        # 随机游走归一化拉普拉斯矩阵: L_rw = I - D^(-1) * A
        degree = np.sum(adj_matrix, axis=1)
        degree_inv = np.power(degree, -1)
        degree_inv[np.isinf(degree_inv)] = 0
        degree_inv = np.diag(degree_inv)
        gso = np.eye(n) - degree_inv @ adj_matrix
        
    elif gso_type == 'sym_renorm_adj':
        # 对称归一化并重新归一化的邻接矩阵: A_sym = D^(-1/2) * A * D^(-1/2)
        degree = np.sum(adj_matrix, axis=1)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
        degree_inv_sqrt = np.diag(degree_inv_sqrt)
        gso = degree_inv_sqrt @ adj_matrix @ degree_inv_sqrt
        
    elif gso_type == 'rw_renorm_adj':
        # 随机游走归一化并重新归一化的邻接矩阵: A_rw = D^(-1) * A
        degree = np.sum(adj_matrix, axis=1)
        degree_inv = np.power(degree, -1)
        degree_inv[np.isinf(degree_inv)] = 0
        degree_inv = np.diag(degree_inv)
        gso = degree_inv @ adj_matrix
        
    else:
        # 默认使用原始邻接矩阵
        gso = adj_matrix
    
    return gso

def calc_chebynet_gso(adj_matrix, gso_type='sym_renorm_adj', K=3):
    """
    计算切比雪夫图卷积的图移位算子
    Args:
        adj_matrix: 邻接矩阵 [n, n]
        gso_type: 归一化类型
        K: 切比雪夫多项式的阶数
    Returns:
        gso: 归一化后的邻接矩阵
    """
    # 首先进行基础归一化
    gso = calc_gso(adj_matrix, gso_type)
    
    # 对于切比雪夫图卷积，需要将特征值缩放到[-1, 1]区间
    # 计算最大特征值
    eigenvals = np.linalg.eigvals(gso)
    max_eigenval = np.max(np.abs(eigenvals))
    
    # 缩放GSO: A_scaled = 2 * A / lambda_max - I
    gso_scaled = 2 * gso / max_eigenval - np.eye(gso.shape[0])
    
    return gso_scaled

def load_adjacency_matrix(file_path, gso_type='sym_renorm_adj', use_chebynet=False, K=3):
    """
    Load and normalize adjacency matrix
    Args:
        file_path: Path to adjacency matrix file
        gso_type: Normalization type
        use_chebynet: Whether to use ChebyNet normalization
        K: ChebyNet polynomial order
    Returns:
        edge_index: Edge indices [2, num_edges]
        edge_weight: Edge weights [num_edges]
    """
    # 读取邻接矩阵
    try:
        # 首先尝试UTF-8编码
        adj_matrix = pd.read_csv(file_path, index_col=0, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            # 如果UTF-8失败，尝试GBK编码
            adj_matrix = pd.read_csv(file_path, index_col=0, encoding='gbk')
        except UnicodeDecodeError:
            # 如果GBK也失败，尝试GB2312编码
            adj_matrix = pd.read_csv(file_path, index_col=0, encoding='gb2312')
    
    # 转换为numpy数组
    adj_matrix = adj_matrix.values
    
    # 根据选择进行归一化
    if use_chebynet:
        adj_matrix = calc_chebynet_gso(adj_matrix, gso_type, K)
    else:
        adj_matrix = calc_gso(adj_matrix, gso_type)
    
    # 获取边的索引和权重
    edge_index = []
    edge_weight = []
    
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] != 0:  # 如果存在连接（包括负值）
                edge_index.append([i, j])
                edge_weight.append(adj_matrix[i, j])
    
    edge_index = torch.LongTensor(edge_index).t()
    edge_weight = torch.FloatTensor(edge_weight)
    
    return edge_index, edge_weight

def load_training_data(file_path, pressure_cols=None, head_loss_cols=None):
    """
    Load training data
    Args:
        file_path: Path to training data file
        pressure_cols: Column names or indices for pressure data
        head_loss_cols: Column names or indices for head loss data
    Returns:
        pressure_data: Pressure data
        head_loss_data: Head loss data
    """
    # Try different encodings to read the data
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            data = pd.read_csv(file_path, encoding='gbk')
        except UnicodeDecodeError:
            data = pd.read_csv(file_path, encoding='gb2312')

    # Adaptively select columns by index or name
    if pressure_cols is None:
        pressure_data = data.iloc[:, 0:33].values  # first 33 columns
    else:
        if isinstance(pressure_cols[0], int):
            pressure_data = data.iloc[:, pressure_cols].values
        else:
            pressure_data = data[pressure_cols].values

    if head_loss_cols is None:
        head_loss_data = data.iloc[:, 33:42].values  # last 9 columns
    else:
        if isinstance(head_loss_cols[0], int):
            head_loss_data = data.iloc[:, head_loss_cols].values
        else:
            head_loss_data = data[head_loss_cols].values

    return pressure_data, head_loss_data

def create_data_loaders(pressure_data, 
                       head_loss_data, 
                       edge_index, 
                       edge_weight,
                       seq_length=12,
                       target_step=12,
                       batch_size=32,
                       train_ratio=0.8,
                       val_ratio=0.1,
                       test_ratio=0.1,
                       random_state=42):
    """
    Create train, validation, and test data loaders
    """
    # 划分数据集
    total_samples = len(pressure_data)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    
    # 训练集
    train_pressure = pressure_data[:train_size]
    train_head_loss = head_loss_data[:train_size]
    
    # 验证集
    val_pressure = pressure_data[train_size:train_size + val_size]
    val_head_loss = head_loss_data[train_size:train_size + val_size]
    
    # 测试集
    test_pressure = pressure_data[train_size + val_size:]
    test_head_loss = head_loss_data[train_size + val_size:]
    
    # 创建数据集
    train_dataset = WaterNetworkDataset(
        train_pressure, train_head_loss, seq_length, target_step
    )
    
    val_dataset = WaterNetworkDataset(
        val_pressure, val_head_loss, seq_length, target_step,
        scaler_pressure=train_dataset.scaler_pressure,
        scaler_head_loss=train_dataset.scaler_head_loss
    )
    
    test_dataset = WaterNetworkDataset(
        test_pressure, test_head_loss, seq_length, target_step,
        scaler_pressure=train_dataset.scaler_pressure,
        scaler_head_loss=train_dataset.scaler_head_loss
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, val_loader, test_loader, train_dataset.scaler_head_loss 