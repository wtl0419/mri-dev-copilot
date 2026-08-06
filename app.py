import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI

st.set_page_config(page_title="MRI Dev-Copilot", page_icon="🧲", layout="wide")

st.title("🧲 MRI Dev-Copilot (磁共振开发与文献 AI 助手)")
st.caption("基于 《Handbook of MRI Pulse Sequences》 知识库 + DeepSeek-V3 引擎")

MY_DEEPSEEK_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"  # 填入你的真实 API Key

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 设置")
    api_key = st.text_input("DeepSeek API Key", type="password", value=MY_DEEPSEEK_KEY)
    st.markdown("---")
    st.markdown("### 📚 当前知识库内容")
    st.info("已加载 65 个章节、936 页专业文献、3260 个知识块。")

# 加载数据库
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

db = load_db()

# 辅助函数：调用 DeepSeek 对英文片段做准确的物理/工程专业中文翻译
def translate_to_chinese(client, english_text):
    try:
        res = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位资深的磁共振（MRI）物理与工程翻译专家。请将以下 MRI 文献英文片段翻译为流畅、严谨的中文。只需输出翻译结果，不要任何多余修饰。"},
                {"role": "user", "content": english_text}
            ],
            temperature=0.1
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"翻译失败: {e}"

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 如果历史消息中包含参考文献，进行渲染
        if "references" in msg:
            with st.expander("📖 查看引用的 Handbook 原文与双语对照"):
                for title, en_text, zh_text in msg["references"]:
                    st.markdown(f"#### {title}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**🇬🇧 英文原文 (Original)**")
                        st.caption(en_text)
                    with col2:
                        st.markdown("**🇨🇳 中文译文 (Translation)**")
                        st.info(zh_text)
                    st.markdown("---")

# 处理用户提问
if prompt := st.chat_input("用中文或英文提问，例如：FSE序列里的破坏梯度应该怎么设置？"):
    current_key = api_key if api_key else MY_DEEPSEEK_KEY
    
    if not current_key or "sk-" not in current_key or "xxxx" in current_key:
        st.error("❌ API Key 无效！请先填入真实的 DeepSeek API Key。")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 1. 检索向量库
    with st.spinner("正在 3260 个知识块中检索并翻译相关文献..."):
        docs = db.similarity_search(prompt, k=3)
        client = OpenAI(api_key=current_key, base_url="https://api.deepseek.com")
        
        context_text = ""
        references = []
        
        for i, doc in enumerate(docs):
            ch = doc.metadata.get('chapter', 'Unk')
            sec = doc.metadata.get('section', 'Unk')
            title = doc.metadata.get('title', 'Unk')
            ref_title = f"[文献{i+1}] 第{ch}章 第{sec}节 - {title}"
            
            en_content = doc.page_content
            # 对检索出的英文片段实时进行专业中文翻译
            zh_content = translate_to_chinese(client, en_content)
            
            references.append((ref_title, en_content, zh_content))
            context_text += f"{ref_title}\n原文内容: {en_content}\n\n"

    # 2. 构造主 Prompt（加入整书大纲，修补上个 Bug！）
    BOOK_TOC = """
    【《Handbook of MRI Pulse Sequences》知识体系地图】：
    - Ch 1-6: 射频脉冲设计 (RF Pulses: SINC, SLR, Composite, Adiabatic 绝热脉冲, Spatial-Spectral)
    - Ch 7-10: 梯度与运动补偿 (Gradients: Readout, Phase Encoding, Crusher, Spoiler, Eddy-Current 涡流, Moment Nulling)
    - Ch 11-13: K空间与图像重建 (k-Space, Bandwidth, 2D/3D, Fourier, Parallel Imaging, Partial Fourier)
    - Ch 14-17: 脉冲序列全家桶 (Spin Echo, Gradient Echo, Inversion Recovery, EPI, RARE/FSE, GRASE, PRESTO, Diffusion, Dixon)
    """

    system_prompt = f"""你是一名资深的磁共振（MRI）软件与序列设计专家。
你掌握整本《Handbook of MRI Pulse Sequences》（共65章，3260个知识块）。

{BOOK_TOC}

请严格根据下方检索到的文献，用严谨专业的中文回答问题。
要求：
1. 若为宏观提问，结合大纲系统回答；若为细节/物理提问，结合检索片段回答。
2. 必须在句末使用 [文献X] 标注引用出处。

【Handbook 检索到的原文段落】：
{context_text}
"""

    # 3. 生成解答与展现双语对照
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
            # 📖 在回答下方，以左右并排（Columns）形式呈现中英双语对照卡片！
            with st.expander("📖 查看引用的 Handbook 原文与同步中文译文", expanded=True):
                for title, en_text, zh_text in references:
                    st.markdown(f"### {title}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**🇬🇧 英文原文 (Original)**")
                        st.caption(en_text)
                    with col2:
                        st.markdown("**🇨🇳 中文译文 (Translation)**")
                        st.info(zh_text)
                    st.markdown("---")

            # 保存到 session_state 保持多轮对话
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "references": references
            })
            
        except Exception as e:
            st.error(f"调用 API 失败: {e}")