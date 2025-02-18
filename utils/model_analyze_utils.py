import torch

def analyze_parameter_memory(model):
    """
    分析模型参数的内存占用
    Args:
        model: PyTorch模型
    """
    def get_size_in_mb(num_params, dtype=torch.float32):
        bytes_per_element = dtype.itemsize
        total_bytes = num_params * bytes_per_element
        return total_bytes / (1024 * 1024)

    print("\n=== Parameter Memory Analysis ===")
    
    # 按模块统计 (修改此部分以避免重复)
    module_stats = {}
    total_size = 0
    counted_params = set()  # 用于追踪已统计的参数
    
    for name, module in model.named_modules():
        if len(list(module.parameters())) > 0:
            module_params = 0
            module_size = 0
            
            # 只统计直接属于该模块的参数
            for param_name, param in module.named_parameters(recurse=False):
                # 创建唯一标识符
                param_id = id(param)
                if param_id not in counted_params:
                    counted_params.add(param_id)
                    num_params = param.numel()
                    module_params += num_params
                    module_size += get_size_in_mb(num_params, param.dtype)
            
            # 只记录有直接参数的模块
            if module_params > 0:
                module_stats[name] = {
                    'parameters': module_params,
                    'size_mb': module_size
                }
                total_size += module_size
    
    # 打印统计结果
    print("\nMemory usage by module:")
    print("-" * 80)
    print(f"{'Module':40s} {'Parameters':>12s} {'Size (MB)':>12s} {'Percentage':>12s}")
    print("-" * 80)
    
    # 按模块大小排序
    sorted_stats = dict(sorted(module_stats.items(), 
                             key=lambda x: x[1]['size_mb'], 
                             reverse=True))
    
    for name, stats in sorted_stats.items():
        percentage = (stats['size_mb'] / total_size) * 100
        print(f"{name:40s} {stats['parameters']:12,d} {stats['size_mb']:12.2f} {percentage:11.2f}%")
    
    print("-" * 80)
    print(f"{'Total':40s} {sum(s['parameters'] for s in module_stats.values()):12,d} {total_size:12.2f} {100:11.2f}%")
    
    for name, stats in module_stats.items():
        percentage = (stats['size_mb'] / total_size) * 100
        print(f"{name:40s} {stats['parameters']:12,d} {stats['size_mb']:12.2f} {percentage:11.2f}%")
    
    print("-" * 80)
    print(f"{'Total':40s} {sum(s['parameters'] for s in module_stats.values()):12,d} {total_size:12.2f} {100:11.2f}%")
    
    # 按参数类型统计
    print("\nMemory usage by parameter type:")
    type_stats = {}
    for name, param in model.named_parameters():
        param_type = param.dtype
        if param_type not in type_stats:
            type_stats[param_type] = {
                'parameters': 0,
                'size_mb': 0
            }
        type_stats[param_type]['parameters'] += param.numel()
        type_stats[param_type]['size_mb'] += get_size_in_mb(param.numel(), param_type)
    
    print("-" * 60)
    print(f"{'Type':20s} {'Parameters':>12s} {'Size (MB)':>12s} {'Percentage':>12s}")
    print("-" * 60)
    
    for dtype, stats in type_stats.items():
        percentage = (stats['size_mb'] / total_size) * 100
        print(f"{str(dtype):20s} {stats['parameters']:12,d} {stats['size_mb']:12.2f} {percentage:11.2f}%")
    
    print("-" * 60)

        # 打印GPU内存使用情况
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


def visualize_deformation_network(model):
    """
    可视化变形网络结构
    Args:
        model: 变形网络模型实例
    """
    print("\n=== Deformation Network Architecture ===")
    
    # 1. 手动打印网络结构
    def print_network_structure(model):
        print("\nLayer Structure:")
        for name, module in model.named_children():
            print(f"\n{name}:")
            print(module)
    
    # 2. 打印示例前向传播
    def test_forward_pass(model):
        print("\nTesting forward pass with sample input:")
        n_points = 1000
        
        # 创建示例输入
        means3D = torch.randn(n_points, 3).cuda()
        scales = torch.randn(n_points, 3).cuda()
        rotations = torch.randn(n_points, 4).cuda()
        opacity = torch.randn(n_points, 1).cuda()
        time = torch.zeros(n_points, 1).cuda()
        
        try:
            with torch.no_grad():
                output = model(means3D, scales, rotations, opacity, time)
                print("\nForward pass successful!")
                print(f"Output shapes:")
                if isinstance(output, tuple):
                    for i, out in enumerate(output):
                        print(f"Output {i}: {out.shape}")
                else:
                    print(f"Output: {output.shape}")
        except Exception as e:
            print(f"\nForward pass failed with error: {str(e)}")
    
    # 3. 打印参数统计
    def print_parameter_stats(model):
        print("\nParameter Statistics:")
        total_params = 0
        trainable_params = 0
        
        for name, param in model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
            print(f"{name}: {param.shape}, Parameters: {num_params:,}")
        
        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    
    # 执行可视化
    print_network_structure(model)
    test_forward_pass(model)
    print_parameter_stats(model)
    analyze_parameter_memory(model)
