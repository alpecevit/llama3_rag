from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.documents import Document
from bs4 import BeautifulSoup as Soup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


url = "https://finance.yahoo.com/news/"
loader = RecursiveUrlLoader(
    url=url, max_depth=2, extractor=lambda x: Soup(x, "html.parser").text
)

# load data
data = loader.load()

# clean documents
parsed_data = []
for doc in data:
    page_content = ' '.join(doc.page_content.strip().split())
    metadata = doc.metadata
    doc_type = doc.type
    document =  Document(page_content=page_content, metadata=metadata, type=doc_type)
    parsed_data.append(document)

# split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(parsed_data)

# sentence transformers
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# create faiss index and save it to local directory
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")
