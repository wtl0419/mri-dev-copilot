import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
from physics.bloch import simulate_spin_echo

# --- 把这里的初始化代码删掉，移到下面的函数内部 ---

def render_simulator():
    """
    重设计后的 3D 序列仿真工作台 (v2.0)
    """
    # ==========================================
    # 🌟 修复关键：将 Session State 初始化移入函数内部！
    # 确保每次用户访问或刷新页面时，都能正确检测并初始化自己的专属状态。
    # ==========================================
    if 'current_frame_index' not in st.session_state:
        st.session_state.current_frame_index = 0
    if 'is_playing' not in st.session_state:
        st.session_state.is_playing = False
    if 'play_speed_ms' not in st.session_state:
        st.session_state.play_speed_ms = 50 # 默认速度，帧间隔50ms
    # 将主界面的标题移至各自的视图内部
    st.title("📉 序列 3D 演化与 PSD 时序工作台")

    # ==========================================
    # 局部侧边栏：仅在此视图激活时显示 (保持不变)
    # ==========================================
    with st.sidebar:
        st.header("⚙️ 模拟与序列配置")
        
        seq_type = st.selectbox(
            "选择 MRI 脉冲序列", 
            ["Spin Echo (SE)", "Gradient Echo (GRE)", "Inversion Recovery (IR)"]
        )
        
        st.subheader("组织物理参数")
        t1 = st.slider("T1 弛豫时间 (ms)", 100, 2000, 800)
        t2 = st.slider("T2 弛豫时间 (ms)", 10, 300, 80)
        
        st.subheader("序列时序参数")
        te = st.slider("TE (回波时间 ms)", 20.0, 100.0, 60.0, step=5.0)
        tr = st.slider("TR (重复时间 ms)", 100, 3000, 1000, step=50)
        
        st.subheader("渲染与显示设置")
        num_iso = st.slider("微观质子数量", 3, 30, 9)
        show_isochromats = st.checkbox("显示微观质子矢量 (Isochromats)", value=True)
        show_main_vector = st.checkbox("显示宏观合矢量 M", value=True)

    # ==========================================
    # 主内容区：物理仿真与同步图表渲染
    # ==========================================
    st.write(f"🌀 当前模式：**{seq_type} - 旋转坐标系与信号波形**")
    
    # 1. 准备仿真数据 (仍然调用物理内核， caching can be added here)
    # 使用 st.cache_data 缓存计算，避免每次 rerun 都重新计算耗时物理模型
    @st.cache_data
    def get_simulation_data(_simulation_func, te_val, t1_val, t2_val, num_iso_val):
        return _simulation_func(TE=te_val, T1=t1_val, T2=t2_val, num_isochromats=num_iso_val, num_steps=80)

    t_array, main_M, iso_M = get_simulation_data(simulate_spin_echo, te, t1, t2, num_iso)
    
    num_frames = len(t_array)
    t_values = t_array
    colors = ["#FF5733", "#33FF57", "#3357FF", "#F39C12", "#9B59B6", "#1ABC9C", "#E74C3C", "#34495E"]

    # 2. 创建 st.empty() 占位符容器，用于在 rerun 时重新绘制图表，提高性能
    charts_container = st.empty()
    control_container = st.empty()
    
    # 获取当前帧索引
    idx = st.session_state.current_frame_index
    # 如果 TR/TE 改变导致 num_frames 改变，确保 idx 在范围内
    if idx >= num_frames:
        idx = num_frames - 1
        st.session_state.current_frame_index = idx

    t_current = t_values[idx]

    # === 基于当前帧 (idx) 构建图表数据 ===
    with charts_container:
        # 3D 旋转坐标系 (静态帧)
        fig3d = go.Figure()
        fig3d.add_trace(go.Scatter3d(x=[-1.2, 1.2], y=[0, 0], z=[0, 0], mode='lines', name="x' (RF)", line=dict(color='gray', width=2)))
        fig3d.add_trace(go.Scatter3d(x=[0, 0], y=[-1.2, 1.2], z=[0, 0], mode='lines', name="y'", line=dict(color='gray', width=2)))
        fig3d.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1.2, 1.2], mode='lines', name="z' (B0)", line=dict(color='gray', width=2)))

        if show_isochromats:
            for j in range(num_iso):
                color = colors[j % len(colors)]
                # 绘制质子点和轨迹线条 (用户图像有线条)
                fig3d.add_trace(go.Scatter3d(
                    x=[0, iso_M[idx, j, 0]], y=[0, iso_M[idx, j, 1]], z=[0, iso_M[idx, j, 2]],
                    mode='lines+markers', line=dict(color=color, width=3), marker=dict(size=[0, 4], color=color), name=f"质子 {j+1}", showlegend=False
                ))

        if show_main_vector:
            fig3d.add_trace(go.Scatter3d(
                x=[0, main_M[idx, 0]], y=[0, main_M[idx, 1]], z=[0, main_M[idx, 2]],
                mode='lines+markers', line=dict(color='red', width=8), marker=dict(size=[0, 8], color='red'), name="主矢量 M", showlegend=False
            ))

        # 去除 Plotly 内置动画按钮和滑块，仅保持布局
        fig3d.update_layout(
            scene=dict(xaxis_range=[-1.2, 1.2], yaxis_range=[-1.2, 1.2], zaxis_range=[-1.2, 1.2], aspectmode='cube'),
            margin=dict(l=0, r=0, b=0, t=30),
            updatemenus=[], # 移除内置按钮
            showlegend=False # 用户图像隐藏了图例
        )

        # 2D PSD PSD PSD图 (静态帧)
        fig_psd = go.Figure()
        m_xy_all = np.sqrt(main_M[:, 0]**2 + main_M[:, 1]**2)
        rf_channel = np.zeros_like(t_array)
        rf_channel[0:max(2, int(num_frames*0.05))] = 0.9
        mid_k = num_frames // 2
        rf_channel[mid_k-1 : mid_k+2] = 1.8

        fig_psd.add_trace(go.Scatter(x=t_array, y=rf_channel, mode='lines', name='RF 脉冲', line=dict(color='blue', width=2)))
        fig_psd.add_trace(go.Scatter(x=t_array, y=m_xy_all, mode='lines', name='横向信号 (M_xy)', line=dict(color='green', width=3)))

        fig_psd.update_layout(
            title="PSD 脉冲波形与信号参考",
            xaxis_title="时间 (ms)", yaxis_title="幅度",
            height=440, margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(tickformat='.1f')
        )
        
        # **核心功能增强：在右侧PSD图上增加同步时间轴指示器垂线**
        fig_psd.add_vline(x=t_current, line_width=2, line_dash="dash", line_color="red", annotation_text=f"t = {t_current:.1f}ms", annotation_position="top")

        # 并排渲染左右双图
        col_3d, col_psd = st.columns([1, 1])
        with col_3d:
            # 使用唯一 key 强制 Plotly Scatter3d 重新渲染数据，解决不重绘问题
            st.plotly_chart(fig3d, use_container_width=True, key=f"3d_chart_frame_{idx}")
        with col_psd:
            st.plotly_chart(fig_psd, use_container_width=True, key=f"psd_chart_frame_{idx}")

    # ==========================================
    # 动画控制与同步时间轴 Slider (新设计)
    # ==========================================
    with control_container:
        st.divider() # 视觉分割线
        st.subheader("动画控制与时间轴")
        
        # 3. 设计外部按钮和控制 Slider 布局
        btn_speed_slider_row = st.columns([1, 1, 1, 1, 4]) # 按钮行: [播放, 暂停, 重置, 空格, 速度滑块]
        
        if btn_speed_slider_row[0].button("▶ 播放丝滑动画", disabled=st.session_state.is_playing):
            st.session_state.is_playing = True
            st.rerun() # 立即 rerun 启动动画循环
        if btn_speed_slider_row[1].button("⏸ 暂停", disabled=not st.session_state.is_playing):
            st.session_state.is_playing = False
            # 无需显式 rerun， st.button 会自动 rerun
        if btn_speed_slider_row[2].button("↺ 重置"):
            st.session_state.is_playing = False
            st.session_state.current_frame_index = 0
            # 无需显式 rerun
            
        # **功能增强：增加慢速播放/速度控制 Slider**
        speed_ms = btn_speed_slider_row[4].slider("播放速度 (ms / 帧)", min_value=10, max_value=200, value=st.session_state.play_speed_ms, step=10, key='speed_slider_control', help="拖动滑块控制动画运行速度。")
        st.session_state.play_speed_ms = speed_ms # 更新速度状态
        
        # 4. 精确时刻选择 Slider (作为主要的时间轴/帧选择器)
        t_current_display = f"时刻: {t_current:.1f}ms"
        selected_idx = st.slider("时间轴 / 精确时刻选择", min_value=0, max_value=num_frames-1, value=st.session_state.current_frame_index, key='frame_slider_control', format="%d", help="拖动滑块精确选择时刻。")
        
        # 如果用户手动拖动了 Slider，更新当前帧索引状态
        if selected_idx != st.session_state.current_frame_index:
            st.session_state.current_frame_index = selected_idx
            # st.slider 会自动触发 rerun，因此图表会自动更新到 selected_idx 的帧

    # ==========================================
    # 动画循环控制逻辑 (在主代码流中，不要放在 with empty() 中)
    # ==========================================
    # 如果正在播放，增加帧
    if st.session_state.is_playing:
        if st.session_state.current_frame_index < num_frames - 1:
            # 增加帧索引
            st.session_state.current_frame_index += 1
            # 强制重新运行以更新图表容器和控制滑块的值
            time.sleep(st.session_state.play_speed_ms / 1000) # time.sleep 控制速度
            st.rerun() # 使用 rerun 触发新的渲染，从而绘制 idx 帧的数据
        else:
            # 播放结束
            st.session_state.is_playing = False 
            st.rerun() # 触发一次 rerun 停止播放并更新按钮状态