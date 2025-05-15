import os
import getpass
import json
from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationChain

app = Flask(__name__)

# 设置环境变量
os.environ.setdefault("USER_AGENT", "MyRAGBot/1.0")
def _set_env(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}:")
_set_env("OPENAI_API_KEY")

# 文件上传目录
UPLOAD_FOLDER = "./docs"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB

# 初始化知识库
docs = []
urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/"
]
doc_splits = None
vectorstore = None
retriever = None
qa_chain = None
conv_chain = None

# 进行知识库的加载和更新
def update_knowledge_base():
    global docs, doc_splits, vectorstore, retriever, qa_chain, conv_chain

    # 加载本地文件和网页
    loader_docx = DirectoryLoader(
        app.config['UPLOAD_FOLDER'], glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader
    )
    docs_local_txt = DirectoryLoader(
        app.config['UPLOAD_FOLDER'], glob="**/*.txt"
    ).load()
    docs_local_docx = loader_docx.load()
    docs = docs_local_txt + docs_local_docx

    # 文本拆分
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=350,
        chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs)

    # 嵌入与向量存储
    os.environ.setdefault("OPENAI_API_BASE", "https://api.siliconflow.cn/v1")
    embeddings = OpenAIEmbeddings(
        model="BAAI/bge-large-zh-v1.5",
        openai_api_base=os.environ["OPENAI_API_BASE"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        chunk_size=32
    )
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 初始化聊天模型和记忆
    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="input",
        return_messages=True
    )
    chat_model = init_chat_model(
        "Qwen/Qwen3-8B",
        model_provider="openai",
        temperature=0,
        openai_api_base=os.environ.get("OPENAI_API_BASE"),
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    # 构建检索问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # 构建带记忆的会话链
    conv_chain = ConversationChain(
        llm=chat_model,
        memory=memory,
        verbose=False,
        input_key="input",
        output_key="output"
    )

# 上传文件API
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    files = request.files.getlist('file')
    filenames = []
    for file in files:
        if file.filename == '':
            return jsonify({'message': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    
    update_knowledge_base()  # 更新知识库
    return jsonify({'message': 'File(s) uploaded successfully', 'filenames': filenames}), 200

# 允许上传的文件类型
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'docx'}

# 聊天接口
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message')
    is_retrieval = data.get('is_retrieval', False)

    if is_retrieval:
        if not qa_chain:
            update_knowledge_base()
        result = qa_chain(user_input)
        answer = result.get("result", "").strip()
        sources = result.get("source_documents", [])
        response = {
            'answer': answer,
            'sources': [src.metadata.get('source') for src in sources]
        }
    else:
        if not conv_chain:
            update_knowledge_base()
        response = conv_chain.invoke({"input": user_input})
        response = {
            'answer': response.get("output")
        }

    return jsonify(response), 200

# 首页路由
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)