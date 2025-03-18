from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os

class ChromaDBManager:
    def __init__(self, pdf_path, chunk_size=250, chunk_overlap=50, persist_directory="./chroma_db"):
        """
        初始化ChromaDB管理器
        
        參數:
            pdf_path (str): PDF文件路徑
            chunk_size (int): 文本區塊大小
            chunk_overlap (int): 區塊重疊大小
            persist_directory (str): 向量庫持久化儲存路徑
        """
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectordb = None
        
    def load_and_process_pdf(self):
        """載入PDF並處理成向量數據庫"""
        # 載入PDF文件
        loader = PyPDFLoader(self.pdf_path)
        
        # 創建文本分割器
        text_splitter = CharacterTextSplitter(
            separator="。",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        # 載入並分割文檔
        pages = loader.load_and_split(text_splitter)
        
        # 創建向量數據庫，並保存到磁碟
        self.vectordb = Chroma.from_documents(pages, self.embeddings, persist_directory=self.persist_directory)
        self.vectordb.persist()
        
        print(f"資料庫已經更新並保存在 {self.persist_directory}")
        return self.vectordb
    
    def refresh_database(self, new_pdf_path=None):
        """刷新資料庫，並可選擇加載新PDF文件
        
        參數:
            new_pdf_path (str): 新的PDF文件路徑，若為None則使用原來的PDF
        """
        if new_pdf_path:
            self.pdf_path = new_pdf_path  # 更新PDF文件路徑
        
        # 重新處理PDF文件並更新向量庫
        self.load_and_process_pdf()
        
        print(f"資料庫已刷新，新的PDF文件為: {self.pdf_path}")

# 用來刷新資料庫的主程式
if __name__ == "__main__":
    
    pdf_path = "data1.pdf"  # PDF 路徑
    chroma_manager = ChromaDBManager(pdf_path=pdf_path)
    
    # 刷新資料庫
    chroma_manager.refresh_database(new_pdf_path="data_new.pdf")  # 傳入新的PDF路徑更新資料庫
