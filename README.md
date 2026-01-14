# RAG-Based-Semantic-Quote-Retrieval-and-Structured-QA-with-Model-Training

## Fine-Tuned Embeddings + FAISS + LLM + Streamlit

---

## 🎯 Objective

The goal of **Task 2** is to design and implement a **Retrieval Augmented Generation (RAG)** system that can:
- Semantically retrieve relevant quotes from a large dataset
- Use a **fine-tuned sentence embedding model**
- Ground LLM responses strictly on retrieved content
- Return **structured JSON outputs**
- Provide an interactive **Streamlit application**

This task demonstrates model training, retrieval, evaluation, and deployment.

---

## 📚 Dataset

- Dataset used: **Abirate/english_quotes**
- Source: HuggingFace  
- Link: https://huggingface.co/datasets/Abirate/english_quotes

### Dataset Fields Used
- `quote`
- `author`
- `tags`

---

## 🧹 Data Preparation

The dataset was preprocessed before model training:
- Lowercasing all text
- Removing null or incomplete entries
- Normalizing quote text
- Preparing query–quote style samples suitable for semantic retrieval

---

## 🔧 Model Fine-Tuning

### Base Model
- Sentence-Transformers (all-MiniLM-L6-v2)

### Fine-Tuning Objective
- Learn semantic similarity between:
  - user queries  
  - quote text  
  - author and tag context

### Why Fine-Tuning?
- Generic embedding models fail to capture:
  - philosophical meaning
  - emotional context
  - author intent

Fine-tuning adapts the model specifically to the **quotes domain**.

### Output
- Fine-tuned embedding model saved locally and reused during inference

---

## 📦 Vector Database

- Vector store: **FAISS**
- Each quote is embedded using the fine-tuned model
- Metadata stored separately:
  - quote
  - author
  - tags

FAISS enables fast similarity search at scale.

---

## 🔄 RAG Pipeline Architecture

1. User submits a natural language query  
2. Query is converted into an embedding  
3. FAISS retrieves top-k most similar quotes  
4. Retrieved quotes are compiled as context  
5. LLM generates a response **only using retrieved context**  
6. Output is returned in structured JSON format  

This ensures **grounded and explainable responses**.

---

## 🤖 LLM Integration

- Model used: **llama-3.3-70b-versatile**
- Provider: **Groq API**

### Why Groq?
- Very fast inference
- Stable structured output support
- Suitable for real-time RAG applications

---

## 📑 Prompt Design

The LLM prompt strictly enforces:
- No external knowledge usage
- Answers only from retrieved quotes
- JSON-only response format

This prevents hallucinations and ensures consistency.

---

## 🧪 RAG Evaluation

RAG evaluation was performed using **one standard framework**:
- RAGAS / Arize Phoenix / Quotient (any one)

### Evaluation Method
- A fixed set of evaluation queries was used, such as:
  - “Quotes about insanity attributed to Einstein”
  - “Motivational quotes tagged accomplishment”
  - “All Oscar Wilde quotes with humor”

### Evaluation Criteria
- Relevance of retrieved quotes
- Faithfulness of LLM response to context
- Completeness of answers

Results and observations are discussed in the evaluation script/notebook.

---

## 🖥 Streamlit Application

An interactive Streamlit application was built with the following features:

### User Capabilities
- Enter natural language queries
- View retrieved quotes with similarity scores
- View LLM-generated summary
- View full structured JSON output
- Download JSON results

This satisfies the **Streamlit deployment requirement** in the assignment.

---

## 🧾 Output Format (Example)

```json
{
  "quotes": [
    {
      "quote": "...",
      "author": "..."
    }
  ],
  "summary": "..."
}
