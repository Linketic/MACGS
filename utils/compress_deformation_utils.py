import torch

class ModelCompressor:
    """模型压缩器 - 物理移除被剪掉的节点"""
    def __init__(self, model):
        self.model = model
        self.pruned_model = None
        
    def _get_valid_indices(self, weights, mask):
        """获取有效权重的索引"""
        mask = mask.bool()
        return torch.nonzero(mask).squeeze()
    
    def compress_layer(self, weights, bias, mask):
        """压缩单层网络"""
        valid_idx = self._get_valid_indices(weights, mask)
        if valid_idx.numel() == 0:
            # 保留最重要的一个神经元
            valid_idx = torch.tensor([torch.argmax(torch.abs(weights).sum(1))], 
                                device=weights.device)
        
        compressed_weights = weights[valid_idx]
        compressed_bias = bias[valid_idx] if bias is not None else None
        
        return compressed_weights, compressed_bias, valid_idx
        
    def compress_network(self, masks):
        """压缩整个网络"""
        compressed_state_dict = {}
        layer_info = {}
        prev_indices = None
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and name in masks:
                # 压缩当前层
                weights = module.weight.data
                bias = module.bias.data if module.bias is not None else None
                
                # 如果有前一层的索引,需要对输入进行压缩
                if prev_indices is not None:
                    weights = weights[:, prev_indices]
                
                # 压缩当前层
                compressed_weights, compressed_bias, curr_indices = self.compress_layer(
                    weights, bias, masks[name]
                )
                
                # 保存压缩后的参数
                compressed_state_dict[f"{name}.weight"] = compressed_weights
                if compressed_bias is not None:
                    compressed_state_dict[f"{name}.bias"] = compressed_bias
                    
                # 记录层信息
                layer_info[name] = {
                    'in_features': weights.size(1),
                    'out_features': len(curr_indices),
                    'indices': curr_indices.cpu()
                }
                
                prev_indices = curr_indices
        
        return compressed_state_dict, layer_info

def save_compressed_model(model_path, deformation_net, pruner, compressor=None):
    """保存压缩后的模型"""
    if compressor is None:
        compressor = ModelCompressor(deformation_net)
    
    # 执行压缩
    compressed_state_dict, layer_info = compressor.compress_network(pruner.masks)
    
    # 收集模型配置
    model_config = {
        'D': deformation_net.D,
        'W': deformation_net.W,
        'input_ch': deformation_net.input_ch,
        'input_ch_time': deformation_net.input_ch_time,
        'grid_pe': deformation_net.grid_pe,
        'skips': deformation_net.skips,
        'args': deformation_net.args
    }
    
    # 保存所有信息
    torch.save({
        'state_dict': compressed_state_dict,
        'layer_info': layer_info,
        'model_config': model_config,
        'masks': pruner.masks
    }, model_path)
    
    return compressed_state_dict, layer_info

def load_compressed_model(model_path, device='cuda'):
    """加载压缩后的模型"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # 解析模型配置
    model_config = checkpoint['model_config']
    layer_info = checkpoint['layer_info']
    
    # 创建压缩后的模型
    from scene.deformation import Deformation
    compressed_model = Deformation(
        D=model_config['D'],
        W=model_config['W'],
        input_ch=model_config['input_ch'],
        input_ch_time=model_config['input_ch_time'],
        grid_pe=model_config['grid_pe'],
        skips=model_config['skips'],
        args=model_config['args']
    )
    
    # 加载压缩后的参数
    compressed_model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    return compressed_model, layer_info