from langchain.document_loaders import TextLoader  #for textfiles
from langchain.text_splitter import CharacterTextSplitter #text splitter
from langchain.embeddings import HuggingFaceEmbeddings #for using HugginFace models
from langchain.vectorstores import FAISS  

from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.document_loaders import UnstructuredPDFLoader  #load pdf
from langchain.indexes import VectorstoreIndexCreator #vectorize db index with chromadb
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredURLLoader  #load urls into docoument-loader
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import os

from langchain.document_loaders import PyPDFLoader
# Load the PDF file from current working directory
loader = PyPDFLoader("contract-data.pdf")
# Split the PDF into Pages
pages = loader.load_and_split()

#import 
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Define chunk size, overlap and separators
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=64,
    separators=['\n\n', '\n', '(?=>\. )', ' ', '']
)
docs  = text_splitter.split_documents(pages)


# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()


#Create the vectorized db
from langchain.vectorstores import FAISS
db = FAISS.from_documents(docs, embeddings)




# llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":1, "max_length":1000000})

# Pass the directory path where the model is stored on your system
model_name = "./google/flan-t5-large"

# Initialize a tokenizer for the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize a model for sequence-to-sequence tasks using the specified pretrained model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)



chain = load_qa_chain(model, chain_type="stuff")

#QUERYING
query = "What is the information related to Rent Increase Decrease in this document?"
docs = db.similarity_search(query)
chain.run(input_documents=docs, question=query)


from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", 
retriever=db.as_retriever(search_kwargs={"k": 3}))


query = "What is the information related to Rent Increase Decrease in this document?"
qa.run(query)