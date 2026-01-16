import torch
import numpy as np
import os
import json
from plyfile import PlyData, PlyElement
from lib.utils.system_utils import mkdir_p

def save_dynamic_actor_ply(actor_model, save_path, metadata=None):
    """
    导出动态 GaussianModelActor 为带有 4D Fourier 参数的自定义 PLY 文件。
    
    Args:
        actor_model: GaussianModelActor 实例
        save_path: .ply 保存路径
        metadata: (Optional) 包含物体元数据(如 track_id, class, extent)的字典，将保存为同名 .json
    """
    mkdir_p(os.path.dirname(save_path))

    # 1. 提取基础几何数据
    xyz = actor_model._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz) # 占位符
    opacities = actor_model._opacity.detach().cpu().numpy()
    scale = actor_model._scaling.detach().cpu().numpy()
    rotation = actor_model._rotation.detach().cpu().numpy()
    
    # 处理语义 (如果有)
    if actor_model._semantic.numel() > 0:
        semantic = actor_model._semantic.detach().cpu().numpy()
    else:
        semantic = None

    # 2. 提取球谐系数 (SH)
    # [Rest] 高频静态分量: [N, (deg+1)**2 - 1, 3] -> Flat [N, n_rest]
    f_rest = actor_model._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

    # [DC] 低频动态 Fourier 分量: 
    # Actor 中的存储形状通常是 [N, Fourier_Dim, 3] (基于转置后的 Parameter)
    # 我们需要将其展平保存。
    f_dc_tensor = actor_model._features_dc.detach()
    fourier_dim = f_dc_tensor.shape[1]
    # 展平为 [N, Fourier_Dim * 3]
    f_dc = f_dc_tensor.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

    # 3. 构建 PLY 属性头 (Header)
    dtype_full = []
    
    # 基础属性
    for attr in ['x', 'y', 'z', 'nx', 'ny', 'nz']:
        dtype_full.append((attr, 'f4'))

    # 4D Fourier DC 属性命名规则: f_dc_{rgb_channel}_{freq_index}
    # 例如: f_dc_0_0 (R通道, 频率0), f_dc_0_1 (R通道, 频率1)...
    for c in range(3): # RGB
        for k in range(fourier_dim):
            dtype_full.append(('f_dc_{}_{}'.format(c, k), 'f4'))

    # 静态 Rest SH 属性命名: f_rest_0 ...
    num_rest = f_rest.shape[1]
    for i in range(num_rest):
        dtype_full.append(('f_rest_{}'.format(i), 'f4'))

    # Opacity, Scale, Rotation
    dtype_full.append(('opacity', 'f4'))
    
    for i in range(scale.shape[1]):
        dtype_full.append(('scale_{}'.format(i), 'f4'))
        
    for i in range(rotation.shape[1]):
        dtype_full.append(('rot_{}'.format(i), 'f4'))

    # Semantic
    if semantic is not None:
        for i in range(semantic.shape[1]):
            dtype_full.append(('semantic_{}'.format(i), 'f4'))

    # 4. 填充数据并写入
    # 拼接所有 numpy 数组
    data_list = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]
    if semantic is not None:
        data_list.append(semantic)
        
    attributes = np.concatenate(data_list, axis=1)
    
    # 创建结构化数组
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    
    # 写入 PLY
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(save_path)
    
    print(f"[Export] Saved dynamic asset: {save_path} (Points: {xyz.shape[0]}, Fourier Dim: {fourier_dim})")

    # 5. 保存元数据 (JSON)
    # 这对于复用非常重要，因为它记录了如何解释 f_dc 以及物体的物理尺寸
    if metadata:
        json_path = save_path.replace('.ply', '.json')
        # 补充模型参数信息
        metadata.update({
            'fourier_dim': fourier_dim,
            'num_points': xyz.shape[0],
            'sh_degree_rest': actor_model.max_sh_degree
        })
        # 转换 numpy 类型为 python 类型以便 json 序列化
        def convert(o):
            if isinstance(o, np.int64): return int(o)
            if isinstance(o, np.float32): return float(o)
            if isinstance(o, torch.Tensor): return o.item()
            return o
            
        with open(json_path, 'w') as f:
            json.dump(metadata, f, default=convert, indent=4)
        print(f"[Export] Saved metadata: {json_path}")