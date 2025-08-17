"""
Wine Analysis RAG System
========================

Flask app that provides RAG-based analysis of wine descriptions from the dataset.
Uses OpenAI embeddings and FAISS for semantic search over wine reviews.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

# LangChain core
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# OpenAI integrations
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Vector store
from langchain_community.vectorstores import FAISS

# Document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")

# Model configuration
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE, api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

@dataclass
class WineConfig:
    data_path: str = ""
    chunk_size: int = 800
    chunk_overlap: int = 100
    sample_size: int = 5000  # Use subset for faster processing

def load_wine_dataset(data_path: str, sample_size: int = 5000):
    """Load wine dataset and convert to documents"""
    try:
        # Read dataset path from file if not provided directly
        if not data_path:
            with open('/Users/jaylonsmith/wine_kaggle/datapath.txt', 'r') as f:
                data_path = f.read().strip().rstrip('.')
        
        df = pd.read_csv(data_path)
        
        # Remove rows with missing descriptions
        df = df.dropna(subset=['description'])
        
        # Sample for faster processing
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        # Convert to LangChain documents
        documents = []
        for idx, row in df.iterrows():
            # Create rich content combining description with metadata
            content = f"Wine Description: {row['description']}\n"
            
            # Add metadata if available
            if pd.notna(row.get('variety')):
                content += f"Variety: {row['variety']}\n"
            if pd.notna(row.get('country')):
                content += f"Country: {row['country']}\n"
            if pd.notna(row.get('province')):
                content += f"Province: {row['province']}\n"
            if pd.notna(row.get('points')):
                content += f"Rating: {row['points']} points\n"
            if pd.notna(row.get('price')):
                content += f"Price: ${row['price']}\n"
            if pd.notna(row.get('winery')):
                content += f"Winery: {row['winery']}\n"
            
            # Create document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    "wine_id": idx,
                    "variety": row.get('variety', 'Unknown'),
                    "country": row.get('country', 'Unknown'),
                    "points": row.get('points', 0),
                    "price": row.get('price', 0),
                    "winery": row.get('winery', 'Unknown')
                }
            )
            documents.append(doc)
        
        return documents
        
    except Exception as e:
        print(f"Error loading wine dataset: {e}")
        return []

class WineAnalyzer:
    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.wine_data_summary = ""
        self.config = WineConfig()
        self.initialize_wine_data()
    
    def initialize_wine_data(self):
        """Initialize the wine analysis system by loading and processing wine data"""
        try:
            # Load wine documents
            raw_docs = load_wine_dataset(self.config.data_path, self.config.sample_size)
            
            if not raw_docs:
                print("No wine data found. Please check the dataset path.")
                self.wine_data_summary = "No wine data found. Please check the dataset path in datapath.txt."
                return
            
            print(f"Loaded {len(raw_docs)} wine reviews")
            
            # Split documents into chunks if needed
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", " ", ""],
            )
            
            docs = text_splitter.split_documents(raw_docs)
            print(f"Created {len(docs)} chunks for analysis")
            
            # Create vector store
            self.vectorstore = FAISS.from_documents(docs, embeddings)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 6})
            
            # Store wine data summary
            self.wine_data_summary = self.create_data_summary(raw_docs)
            
            # Setup RAG chain
            self.setup_rag_chain()
            
        except Exception as e:
            print(f"Error initializing wine data: {e}")
            self.wine_data_summary = f"Error loading wine data: {str(e)}"
    
    def create_data_summary(self, docs):
        """Create a summary of the wine dataset for display"""
        if not docs:
            return "No wine data available"
        
        # Extract some statistics
        varieties = set()
        countries = set()
        points = []
        prices = []
        
        for doc in docs[:100]:  # Sample first 100 for summary
            meta = doc.metadata
            if meta.get('variety') != 'Unknown':
                varieties.add(meta.get('variety'))
            if meta.get('country') != 'Unknown':
                countries.add(meta.get('country'))
            if meta.get('points', 0) > 0:
                points.append(meta.get('points'))
            if meta.get('price', 0) > 0:
                prices.append(meta.get('price'))
        
        summary = f"""🍷 Wine Dataset Summary
        
📊 Total Reviews: {len(docs)}
🍇 Varieties: {len(varieties)} different types
🌍 Countries: {len(countries)} different countries
⭐ Rating Range: {min(points) if points else 'N/A'} - {max(points) if points else 'N/A'} points
💰 Price Range: ${min(prices) if prices else 'N/A'} - ${max(prices) if prices else 'N/A'}

Top Varieties: {', '.join(list(varieties)[:5])}
Top Countries: {', '.join(list(countries)[:5])}

You can ask questions like:
• "What are the characteristics of highly rated wines?"
• "Compare Pinot Noir wines from different regions"
• "What makes a wine expensive?"
• "Find wines similar to Cabernet Sauvignon"
"""
        return summary
    
    def format_docs(self, docs):
        """Format documents for RAG context"""
        formatted = []
        for d in docs:
            variety = d.metadata.get('variety', 'Unknown')
            country = d.metadata.get('country', 'Unknown')
            points = d.metadata.get('points', 'N/A')
            formatted.append(f"[Wine: {variety} from {country}, {points} points]\n{d.page_content}")
        return "\n\n---\n\n".join(formatted)
    
    def setup_rag_chain(self):
        """Setup the RAG chain for wine analysis"""
        if not self.retriever:
            return
            
        RAG_PROMPT = PromptTemplate.from_template(
            """You are a wine expert and sommelier. Use the provided wine reviews and data to answer questions about wines.
Provide detailed, knowledgeable responses about wine characteristics, regions, varieties, and recommendations.
When relevant, cite specific wines or patterns you see in the data.

Question:
{question}

Wine Data Context:
{context}

Provide a detailed analysis with specific examples from the wine data when possible. Include recommendations if appropriate."""
        )
        
        self.rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | RAG_PROMPT
            | llm
            | StrOutputParser()
        )
    
    def get_wine_summary(self):
        """Return the wine dataset summary for display"""
        return self.wine_data_summary
    
    def analyze_wines(self, query):
        """Analyze wines based on user query using RAG"""
        if not self.rag_chain:
            return "Wine analysis system not properly initialized. Please check the dataset path."
        
        try:
            response = self.rag_chain.invoke(query)
            return response
        except Exception as e:
            return f"Error processing your wine question: {str(e)}"
    
    def search_similar_wines(self, description, k=5):
        """Find wines similar to a given description"""
        if not self.vectorstore:
            return "Wine search not available. Please check the dataset."
        
        try:
            docs = self.vectorstore.similarity_search(description, k=k)
            results = []
            for doc in docs:
                meta = doc.metadata
                results.append({
                    'variety': meta.get('variety', 'Unknown'),
                    'country': meta.get('country', 'Unknown'),
                    'points': meta.get('points', 'N/A'),
                    'price': meta.get('price', 'N/A'),
                    'winery': meta.get('winery', 'Unknown'),
                    'description': doc.page_content[:200] + "..."
                })
            return results
        except Exception as e:
            return f"Error searching wines: {str(e)}"

# Initialize the wine analyzer
wine_analyzer = WineAnalyzer()

app = Flask(__name__, static_folder='frontend/build', static_url_path='')
CORS(app)

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Process the message and get wine analysis
        response = wine_analyzer.analyze_wines(message)
        
        return jsonify({
            'response': response,
            'timestamp': '2024-01-01T00:00:00Z'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wines/summary', methods=['GET'])
def get_wine_summary():
    try:
        summary = wine_analyzer.get_wine_summary()
        
        return jsonify({
            'summary': summary,
            'timestamp': '2024-01-01T00:00:00Z'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wines/search', methods=['POST'])
def search_wines():
    try:
        data = request.get_json()
        description = data.get('description', '')
        k = data.get('limit', 5)
        
        if not description:
            return jsonify({'error': 'No description provided'}), 400
        
        results = wine_analyzer.search_similar_wines(description, k)
        
        return jsonify({
            'results': results,
            'timestamp': '2024-01-01T00:00:00Z'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def serve():
    return app.send_static_file('index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return app.send_static_file(path)

if __name__ == '__main__':
    print("Starting Wine Analysis Server...")
    print(f"Dataset path: {wine_analyzer.config.data_path}")
    app.run(debug=True, host='0.0.0.0', port=5001)
