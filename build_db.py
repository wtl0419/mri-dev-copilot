import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. 填入你真实的 PDF 存放绝对路径
PDF_DIR = r"D:\RAG\MRI\handbook-of-mri-pulse-sequences"
DB_DIR = "./chroma_db"  # 生成的数据库直接保存在当前 py_test 项目下

def main():
    print("正在加载 Embeddings 模型 (首次运行会自动下载轻量级模型)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    documents = []
    if not os.path.exists(PDF_DIR):
        print(f"❌ 找不到路径: {PDF_DIR}，请检查路径是否正确！")
        return

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    print(f"找到 {len(pdf_files)} 个 PDF 章节文件，开始读取...")

    for file_name in pdf_files:
        file_path = os.path.join(PDF_DIR, file_name)
        
        # 匹配文件名格式: 第X章-第X.Y节-英文名.pdf
        match = re.search(r"第(\d+)章-第([\d\.]+)节-(.+)\.pdf", file_name)
        chapter = match.group(1) if match else "Unk"
        section = match.group(2) if match else "Unk"
        title = match.group(3) if match else file_name

        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            for p in pages:
                p.metadata["chapter"] = chapter
                p.metadata["section"] = section
                p.metadata["title"] = title
                p.metadata["file_name"] = file_name
                documents.append(p)
        except Exception as e:
            print(f"⚠️ 读取文件 {file_name} 出错: {e}")

    print(f"所有 PDF 读取完毕，共加载 {len(documents)} 页数据。")

    # 按最佳颗粒度切分文本
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    print(f"文本切分完成！共生成 {len(chunks)} 个知识 Chunk。")

    # 写入本地 Chroma 数据库
    print("正在构建向量数据库 (ChromaDB)，请稍候...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print("✅ 向量数据库构建成功！数据已保存在当前项目下的 ./chroma_db 文件夹中")

if __name__ == "__main__":
    main()