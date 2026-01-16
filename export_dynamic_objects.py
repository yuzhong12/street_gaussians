import torch
import os
import sys
from argparse import ArgumentParser

# 引入项目库
from lib.config import cfg
from lib.utils.cfg_utils import load_cfg
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.datasets.dataset import Dataset
from lib.utils.system_utils import searchForMaxIteration
from lib.utils.general_utils import safe_state
from lib.utils.asset_exporter import save_dynamic_actor_ply # 引入刚才写好的接口

def export_assets(args):
    # 1. 加载配置
    cfg.model_path = args.model_path
    load_cfg(args.config)
    
    # 强制不使用 dataloader 的多进程，防止导出时资源占用过大
    cfg.data.num_workers = 0 
    
    print(f"Loading model from: {args.model_path}")
    
    # 2. 初始化模型和场景
    # 注意：我们只需要 dataset 来获取 metadata，不需要加载图像数据
    dataset = Dataset() 
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    
    # 3. 加载 Checkpoint
    if args.iteration == -1:
        loaded_iter = searchForMaxIteration(os.path.join(args.model_path, "chkpnt"))
    else:
        loaded_iter = args.iteration
        
    ckpt_path = os.path.join(args.model_path, "chkpnt", f'iteration_{loaded_iter}.pth')
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading checkpoint state: {ckpt_path}")
    state_dict = torch.load(ckpt_path)
    gaussians.load_state_dict(state_dict)
    
    # 4. 创建导出目录
    export_dir = os.path.join(args.model_path, "exported_assets")
    os.makedirs(export_dir, exist_ok=True)
    
    # 5. 遍历并导出所有动态物体
    print(f"Start exporting dynamic objects...")
    count = 0
    
    # graph_obj_list 可能需要在 parse_camera 后才生成，或者直接访问 obj_list
    # 我们直接访问在 setup_functions 中注册的 obj_list
    if not hasattr(gaussians, 'obj_list') or len(gaussians.obj_list) == 0:
        print("No dynamic objects found in the model.")
        return

    for obj_name in gaussians.obj_list:
        # 获取 Actor 模型实例
        actor_model = getattr(gaussians, obj_name)
        
        # 准备保存路径
        ply_name = f"{obj_name}.ply"
        save_path = os.path.join(export_dir, ply_name)
        
        # 准备元数据 (用于后续重建 Asset)
        # 从 Actor 实例中提取关键物理属性
        meta = {
            'obj_name': obj_name,
            'class_name': actor_model.obj_class,
            'track_id': actor_model.track_id,
            'bbox': actor_model.bbox.tolist(), # [Length, Width, Height]
            'extent': actor_model.extent.item(), # 这里的 extent 是缩放归一化参数
            'fourier_scale': actor_model.fourier_scale
        }
        
        # 调用接口进行保存
        save_dynamic_actor_ply(actor_model, save_path, metadata=meta)
        count += 1
        
    print(f"\n[Done] Successfully exported {count} assets to {export_dir}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Export dynamic Street Gaussian objects to assets")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model output folder")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file (e.g., configs/waymo/seq_xxx.yaml)")
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration to load (default: max)")
    
    args = parser.parse_args()
    
    # 初始化环境
    safe_state(True)
    
    export_assets(args)