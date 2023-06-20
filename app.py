
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, jsonify
from flask import request
import json
import time
import requests
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import VectorDBQA, OpenAI
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
import pinecone
# import nltk
# nltk.download('punkt')
# from nltk.tokenize import word_tokenize
from collections import namedtuple
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the named tuple structure
Document = namedtuple('Document', ['page_content', 'metadata'])

def read_text_file(url):
    response = requests.get(url)
    print(response.text)
    if response.status_code == 200:
        # Extract the content as a single string
        content = response.text.strip()

        # Create the Document namedtuple and populate the array
        documents = [Document(page_content=content, metadata={'source': 'data.txt'})]
        return documents
    else:
        print("Error: Failed to fetch the text file.")
        return []

pinecone.init(
    api_key="31b08bd6-3937-487d-803e-4eee036a8aaa",
    environment="us-west4-gcp-free",
)

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello from Flask!'

@app.route("/testpost", methods=['POST'])
def testPost():
    data = request.get_json()

    # Retrieve values
    question = data.get("question")
    file_url = data.get("file_url")

    json_response = {"question": question, "file_url": file_url, "message": "POST request received"}

    return jsonify(json_response)

@app.route('/get-data', methods=['POST'])
def paconv():
    # Access the JSON payload
    data = request.get_json()

    print("API KEY: ", os.environ.get("OPENAI_API_KEY"))

    # Retrieve values
    question = data.get("question")
    file_url = data.get("file_url")

    print("FILE URL: ", file_url)

    document = read_text_file(file_url)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="pythonllm-embeddings2"
    )

    qa = VectorDBQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", vectorstore=docsearch, return_source_documents=True
    )

    query = question
    result = qa({"query": query})

    json_response = {"question": question, "file_url": file_url, "message": "POST request received"}

    print("RESULT: ", result)
    print("RESULT TYPE: ", type(result))


    return jsonify(json_response)


if __name__ == "__main__":
    app.run(debug=True)
