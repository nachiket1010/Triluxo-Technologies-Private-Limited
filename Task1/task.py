from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import SimpleQAChain

# Step 1: Extract data from URL
loader = WebBaseLoader(urls=["https://brainlox.com/courses/category/technical"])
docs = loader.load()

# Step 2: Create embeddings and store in a vector store
embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
texts = [doc.content for doc in docs]
embeddings = embedder.embed_documents(texts)
vector_store = FAISS.from_embeddings(embeddings, texts)

# Step 3: Create a Flask RESTful API
app = Flask(__name__)
api = Api(app)

class Chatbot(Resource):
    def post(self):
        data = request.get_json()
        query = data.get("query")
        
        # Create a QA chain
        chain = SimpleQAChain(vector_store=vector_store, embedder=embedder)
        
        # Get the answer from the QA chain
        answer = chain.run(input=query)
        
        return jsonify({"answer": answer})

api.add_resource(Chatbot, '/chat')

if __name__ == '__main__':
    app.run(debug=True)
