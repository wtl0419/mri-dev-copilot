import streamlit as st

def render_chatbot():
    """
    渲染 RAG 理论问答助手工作台
    此视图不会向侧边栏注入任何参数，保持页面清爽。
    """
    st.title("💬 Handbook AI 理论问答助手")
    st.caption("独立运行的文献知识库接口。可以在此输入代码挂载你的 RAG 查询逻辑。")
    
    # 初始化隔离的状态机
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 循环渲染历史对话内容
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 捕获输入与处理推理
    if prompt := st.chat_input("在此处输入需要查询的 MRI 物理问题或序列原理..."):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 预留的 RAG 核心处理区域
        with st.chat_message("assistant"):
            response = f"🤖 已收到你的问题：**【{prompt}】**。\n\n*(提示：你的 RAG 检索管道可以直接对接在这个代码块内部，输出将自动更新在这里)*"
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})