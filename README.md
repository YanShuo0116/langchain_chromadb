## 基於LangChain和ChromaDB讀取PDF文件使用gimini。

## 安裝

```bash
pip install pypdf
pip install chromadb
pip install google.generativeai
pip install langchain-google-genai
pip install langchain
pip install langchain_community
pip install jupyter
```

### 設置API金鑰

在終端機中設置Google API金鑰：

```bash
export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
```

## 使用

1. 將PDF檔案命名為`data.pdf`放置於專案目錄
2. 執行程式:
```bash
python app.py
```
3.使用chroma.py下方函數刷新pdf
```bash
python chroma.py 
```


## 自訂

- 修改`app.py`中的問題文字以查詢不同內容
- 調整`chunk_size`和`chunk_overlap`參數以優化分析效果
