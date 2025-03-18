from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma

# 載入模型
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# PDF文件分塊
loader = PyPDFLoader("data.pdf")
text_splitter = CharacterTextSplitter(
    separator="。",
    chunk_size=250,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)
pages = loader.load_and_split(text_splitter)

# 建立向量庫
vectordb = Chroma.from_documents(pages, embeddings)

# 設定檢索器
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# 創建檢索鏈
template = """
你是個有幫助的AI助手。
請根據提供的上下文來回答問題。
context: {context}
input: {input}
answer:
"""
prompt = PromptTemplate.from_template(template)
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# 提問與答覆
response = retrieval_chain.invoke({"input": "簡述台灣歷史各時期語言政策"})

# 輸出
print("回答:"+response["answer"])
