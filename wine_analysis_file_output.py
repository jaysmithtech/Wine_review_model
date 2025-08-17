"""
Wine Analysis File Output System
===============================

Generates wine analysis outputs and saves them to files in the directory.
Uses RAG to analyze wine descriptions and saves results as text files.
"""

import pandas as pd
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Optional

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

def load_wine_dataset(data_path: str = "", sample_size: int = 3000):
    """Load wine dataset and convert to documents"""
    try:
        # Read dataset path from file if not provided directly
        if not data_path:
            with open('/Users/jaylonsmith/wine_kaggle/datapath.txt', 'r') as f:
                data_path = f.read().strip().rstrip('.')
        
        print(f"Loading wine dataset from: {data_path}")
        df = pd.read_csv(data_path)
        
        # Remove rows with missing descriptions
        df = df.dropna(subset=['description'])
        
        # Sample for faster processing
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            print(f"Using sample of {sample_size} wines for analysis")
        
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
        
        print(f"Created {len(documents)} wine documents")
        return documents, df
        
    except Exception as e:
        print(f"Error loading wine dataset: {e}")
        return [], pd.DataFrame()

class WineAnalysisFileGenerator:
    def __init__(self, output_dir="./wine_analysis_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.df = None
        
    def initialize_system(self):
        """Initialize the wine analysis system"""
        print("Initializing wine analysis system...")
        
        # Load wine data
        documents, self.df = load_wine_dataset()
        if not documents:
            print("Failed to load wine data")
            return False
        
        # Create chunks if needed
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""],
        )
        
        docs = text_splitter.split_documents(documents)
        print(f"Created {len(docs)} chunks for analysis")
        
        # Create vector store
        print("Creating vector embeddings...")
        self.vectorstore = FAISS.from_documents(docs, embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 6})
        
        # Setup RAG chain
        self.setup_rag_chain()
        print("System initialized successfully!")
        return True
    
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
    
    def analyze_and_save(self, query: str, filename: str):
        """Analyze wine data based on query and save to markdown file"""
        if not self.rag_chain:
            return "System not initialized"
        
        try:
            print(f"Analyzing: {query}")
            response = self.rag_chain.invoke(query)
            
            # Save to markdown file
            output_file = self.output_dir / f"{filename}.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# {query}\n\n")
                f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")
                f.write(response)
            
            print(f"Analysis saved to: {output_file}")
            return response
            
        except Exception as e:
            error_msg = f"Error analyzing wines: {str(e)}"
            print(error_msg)
            return error_msg
    
    def generate_dataset_summary(self):
        """Generate and save dataset summary as markdown"""
        if self.df is None or self.df.empty:
            return "No dataset loaded"
        
        # Calculate statistics
        total_wines = len(self.df)
        varieties = self.df['variety'].value_counts().head(10)
        countries = self.df['country'].value_counts().head(10)
        avg_points = self.df['points'].mean()
        avg_price = self.df['price'].mean()
        price_range = f"${self.df['price'].min():.0f} - ${self.df['price'].max():.0f}"
        points_range = f"{self.df['points'].min()} - {self.df['points'].max()}"
        
        summary = f"""# Wine Dataset Summary

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Dataset Overview

- **Total wine reviews:** {total_wines:,}
- **Average rating:** {avg_points:.1f} points
- **Average price:** ${avg_price:.2f}
- **Price range:** {price_range}
- **Points range:** {points_range}

## Top 10 Wine Varieties

| Variety | Count |
|---------|-------|
"""
        
        for variety, count in varieties.items():
            summary += f"| {variety} | {count} |\n"
        
        summary += f"\n## Top 10 Countries\n\n| Country | Count |\n|---------|-------|\n"
        for country, count in countries.items():
            summary += f"| {country} | {count} |\n"
        
        # Save summary as markdown
        summary_file = self.output_dir / "dataset_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"Dataset summary saved to: {summary_file}")
        return summary
    
    def find_similar_wines(self, description: str, k: int = 5):
        """Find and save similar wines based on description as markdown"""
        if not self.vectorstore:
            return "Vector store not initialized"
        
        try:
            docs = self.vectorstore.similarity_search(description, k=k)
            
            results = f"""# Similar Wine Search

**Query:** {description}  
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Similar Wines Found

"""
            
            for i, doc in enumerate(docs, 1):
                meta = doc.metadata
                results += f"### {i}. {meta.get('variety', 'Unknown')} from {meta.get('country', 'Unknown')}\n\n"
                results += f"- **Rating:** {meta.get('points', 'N/A')} points\n"
                results += f"- **Price:** ${meta.get('price', 'N/A')}\n"
                results += f"- **Winery:** {meta.get('winery', 'Unknown')}\n"
                results += f"- **Description:** {doc.page_content[:200]}...\n\n"
            
            # Save results as markdown
            search_file = self.output_dir / "similar_wines_search.md"
            with open(search_file, 'w', encoding='utf-8') as f:
                f.write(results)
            
            print(f"Similar wines search saved to: {search_file}")
            return results
            
        except Exception as e:
            error_msg = f"Error searching wines: {str(e)}"
            print(error_msg)
            return error_msg

def main():
    """Main function to generate wine analysis outputs"""
    print("Wine Analysis File Output Generator")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = WineAnalysisFileGenerator()
    
    if not analyzer.initialize_system():
        print("Failed to initialize system. Exiting.")
        return
    
    # Generate dataset summary
    print("\n1. Generating dataset summary...")
    analyzer.generate_dataset_summary()
    
    # Predefined analysis queries
    analyses = [
        ("What are the characteristics of highly rated wines (90+ points)?", "highly_rated_wines"),
        ("Compare Pinot Noir wines from different regions", "pinot_noir_regional_comparison"),
        ("What makes expensive wines different from affordable ones?", "expensive_vs_affordable_wines"),
        ("Analyze the characteristics of French wines vs Italian wines", "french_vs_italian_wines"),
        ("What are the common descriptors for excellent Cabernet Sauvignon?", "cabernet_sauvignon_analysis"),
        ("Identify patterns in wine descriptions for different price ranges", "price_range_patterns"),
        ("What regions produce the most highly rated Chardonnay?", "chardonnay_regional_analysis"),
        ("Analyze the relationship between wine variety and typical flavor profiles", "variety_flavor_profiles")
    ]
    
    # Generate analyses
    print(f"\n2. Generating {len(analyses)} wine analyses...")
    for i, (query, filename) in enumerate(analyses, 1):
        print(f"   {i}/{len(analyses)}: {filename}")
        analyzer.analyze_and_save(query, filename)
    
    # Generate similar wine search
    print("\n3. Generating similar wine search...")
    search_description = "Rich, full-bodied red wine with dark fruit flavors, oak aging, and smooth tannins"
    analyzer.find_similar_wines(search_description)
    
    print(f"\n✅ All analyses complete!")
    print(f"📁 Output files saved to: {analyzer.output_dir}")
    print("\nGenerated files:")
    for file in analyzer.output_dir.glob("*.md"):
        print(f"   - {file.name}")

if __name__ == "__main__":
    main()
