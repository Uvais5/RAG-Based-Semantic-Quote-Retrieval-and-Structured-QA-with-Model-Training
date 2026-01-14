import streamlit as st
import pandas as pd
import json
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Semantic Quote Explorer", page_icon="📜")

@st.cache_resource
def load_rag_assets():
    
    
    model = SentenceTransformer('fine_tuned_qoute-retriever')
    index = faiss.read_index("vector_databse/quotes_vector_db.faiss")
    metadata = pd.read_pickle("vector_databse/quotes_metadata.pkl")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return model, index, metadata, client

model, index, metadata, groq_client = load_rag_assets()


def get_rag_response(query):
    
    query_vec = model.encode([query]).astype('float32')
    distances, indices = index.search(query_vec, k=3)
    
    retrieved_quotes = []
    context_text = ""
    for i, idx in enumerate(indices[0]):
        row = metadata.iloc[idx]
        quote_info = {
            "quote": row['quote'],
            "author": row['author'],
            "tags": row['tags'],
            "score": float(distances[0][i])
        }
        retrieved_quotes.append(quote_info)
        context_text += f"Quote: {row['quote']} | Author: {row['author']}\n"

    
    prompt = f"""
    Answer the user query based ONLY on the provided quotes. 
    Query: {query}
    Context:
    {context_text}
    
    Return the response in a strict JSON format with the following keys:
    'quotes': a list of the retrieved quotes and authors
    'summary': a 2-sentence explanation of how these quotes relate to the query.
    """
    
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile", 
        response_format={"type": "json_object"}
    )
    
    return json.loads(chat_completion.choices[0].message.content), retrieved_quotes


st.title(" Semantic Quote Retrieval")
st.write("Retrieve quotes from the Abirate/english_quotes dataset using RAG.")


user_query = st.text_input("Search for quotes:", placeholder="e.g., Show me quotes about courage by women authors", key="query_input")

submit_button = st.button("Enter")

if submit_button and user_query:
    with st.spinner("Retrieving wisdom..."):
        try:
            
            json_response, source_data = get_rag_response(user_query)
            

            st.subheader("🔍 Source Quotes (Retrieved)")
            for src in source_data:
                with st.expander(f"Author: {src['author']} | Similarity Score: {src['score']:.4f}"):
                    st.write(f"\"{src['quote']}\"")
                    st.caption(f"Tags: {', '.join(src['tags'])}")
            
            
            st.subheader("LLM Response")
            st.info(f"**Summary:** {json_response.get('summary', 'No summary generated.')}")
            
            
            st.subheader("📄 Structured JSON Response")
            st.json(json_response)
            
            
            json_str = json.dumps(json_response, indent=4)
            st.download_button(
                label="Download JSON Response",
                data=json_str,
                file_name="quote_response.json",
                mime="application/json"
            )
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")


with st.sidebar:
    st.header("Project Info")
    st.write("Using Fine-tuned MiniLM + Groq Llama-3")
    st.divider()
    st.write("🔗 [HuggingFace Dataset](https://huggingface.co/datasets/Abirate/english_quotes)")
