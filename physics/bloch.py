import numpy as np

def simulate_spin_echo(TE: float, T1: float, T2: float, num_isochromats: int = 15, num_steps: int = 100):
    """
    自旋回波 (Spin Echo) 严格物理演化：
    t=0: 90°x 激发 -> FID 信号衰减
    t=TE/2: 180°x 重聚焦脉冲 -> 相角翻转
    t=TE: 质子重聚 -> 产生 Spin Echo 回波
    """
    t_array = np.linspace(0, TE, num_steps)
    
    # 模拟 B0 不均匀性造成的频偏范围 (rad/ms)
    offsets = np.linspace(-0.25, 0.25, num_isochromats)
    
    # 存储轨迹: (num_steps, num_isochromats, 3)
    iso_M = np.zeros((num_steps, num_isochromats, 3))
    
    # 初始状态: 全在 z' 轴上 (0, 0, 1)
    iso_M[0, :, :] = np.array([0.0, 0.0, 1.0])
    
    rf90_frames = max(2, int(num_steps * 0.06))
    rf180_frames = max(2, int(num_steps * 0.06))
    rf180_center_idx = num_steps // 2  # TE / 2 时刻
    
    for i in range(1, num_steps):
        dt = t_array[i] - t_array[i-1]
        prev_M = iso_M[i-1, :, :].copy()
        
        # --- 1. 90°x RF 脉冲 (把 Mz 压倒到 y' 轴) ---
        if i <= rf90_frames:
            d_theta = (np.pi / 2) / rf90_frames
            R_x = np.array([
                [1, 0, 0],
                [0, np.cos(d_theta), np.sin(d_theta)],
                [0, -np.sin(d_theta), np.cos(d_theta)]
            ])
            for j in range(num_isochromats):
                iso_M[i, j] = R_x @ prev_M[j]
            continue

        # --- 2. 180°x RF 脉冲 (t = TE/2 绕 x' 轴旋转 180°，翻转相角) ---
        if abs(i - rf180_center_idx) <= (rf180_frames // 2):
            d_theta = np.pi / rf180_frames
            R_x = np.array([
                [1, 0, 0],
                [0, np.cos(d_theta), np.sin(d_theta)],
                [0, -np.sin(d_theta), np.cos(d_theta)]
            ])
            for j in range(num_isochromats):
                iso_M[i, j] = R_x @ prev_M[j]
            continue

        # --- 3. 自由进动与 T1/T2 弛豫 (进动包含相角积累) ---
        for j in range(num_isochromats):
            dw = offsets[j]
            x_prev, y_prev, z_prev = prev_M[j]
            
            # 旋转坐标系下由 dw 引起的自由进动
            x_new = (x_prev * np.cos(dw * dt) + y_prev * np.sin(dw * dt)) * np.exp(-dt / T2)
            y_new = (-x_prev * np.sin(dw * dt) + y_prev * np.cos(dw * dt)) * np.exp(-dt / T2)
            z_new = 1.0 - (1.0 - z_prev) * np.exp(-dt / T1)
            
            iso_M[i, j] = [x_new, y_new, z_new]

    # 计算宏观合矢量 M = 所有质子的矢量和
    main_M = np.mean(iso_M, axis=1)
    
    return t_array, main_M, iso_M