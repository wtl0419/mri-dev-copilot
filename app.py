import streamlit as st

# 从视图模块中导入渲染接口
from views.simulator import render_simulator
from views.chatbot import render_chatbot

# ==========================================
# 1. 页面全局配置 (必须在脚本第一行调用)
# ==========================================
st.set_page_config(
    page_title="MRI Dev-Copilot", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. 侧边栏：全局导航菜单
# ==========================================
with st.sidebar:
    st.title("🧲 导航菜单")
    # 使用单选按钮 (radio) 替代 tabs，作为页面的核心切换器
    app_mode = st.radio(
        "选择工作区",
        ["📉 序列仿真工作台", "💬 AI 理论问答助手"],
        label_visibility="collapsed" # 隐藏多余的标签，让 UI 更简洁
    )
    
    # 增加一条视觉分割线，将导航栏与下方的动态参数面板隔开
    st.divider()

# ==========================================
# 3. 核心功能区：条件路由分发
# ==========================================
# 根据用户在侧边栏的选择，动态加载对应的视图函数
# 只有在 render_simulator() 被调用时，它的专属参数才会出现在侧边栏下方
if app_mode == "📉 序列仿真工作台":
    render_simulator()
elif app_mode == "💬 AI 理论问答助手":
    render_chatbot()