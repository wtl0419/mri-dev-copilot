from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI

# 1. 填入你的真实 API Key (例如 "sk-xxxxxxxxxxxxxxxxxxxxxxxx")
DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"

client = OpenAI(
    api_key=DEEPSEEK_API_KEY, 
    base_url="https://api.deepseek.com"
)

print("正在加载本地 Handbook 向量数据库...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def ask_mri_copilot(question: str):
    print(f"\n【用户提问】: {question}")
    print("正在 3260 个 Handbook 知识块中检索相关内容...")
    
    # 检索最相关的 3 个段落
    docs = db.similarity_search(question, k=3)
    
    context_text = ""
    print("\n" + "-"*15 + " 检索到的文献片段 " + "-"*15)
    for i, doc in enumerate(docs):
        ch = doc.metadata.get('chapter', 'Unk')
        sec = doc.metadata.get('section', 'Unk')
        title = doc.metadata.get('title', 'Unk')
        
        ref = f"[文献{i+1}] 第{ch}章 第{sec}节 - {title}"
        print(f"📌 {ref}")
        context_text += f"{ref}\n原文内容: {doc.page_content}\n\n"
    print("-"*40)

    # 构造 Prompt 注入检索到的上下文
    system_prompt = f"""你是一名资深的磁共振（MRI）软件与序列设计专家。
请严谨地根据下方《Handbook of MRI Pulse Sequences》的检索内容来回答问题。
要求：
1. 回答要严谨专业，符合物理与工程事实。
2. 引用检索内容时，必须在句末使用 [文献X] 格式标出出处。

【Handbook 权威检索段落】：
{context_text}
"""

    print("\n正在调用 DeepSeek 生成解答...")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.3
    )
    
    print("\n" + "="*20 + " DeepSeek 专家解答 " + "="*20)
    print(response.choices[0].message.content)
    print("="*50)

if __name__ == "__main__":
    # 直接用中文提问！
    test_q = "梯度回波序列里的扰相梯度（Spoiler Gradients）主要是做什么用的？极性怎么设置？"
    ask_mri_copilot(test_q)