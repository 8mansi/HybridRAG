# pip install sentence_transformers
# !pip install requests beautifulsoup4 
# !pip install tiktoken
# pip install faiss-cpu
# pip install rank-bm25
# pip install streamlit
# pip install transformers torch
# pip install pandas
# python -m streamlit run .\HybridRag.py

import pandas as pd
import requests, json, time
from bs4 import BeautifulSoup
import tiktoken
import uuid
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st
tokenizer = tiktoken.get_encoding("cl100k_base")

class WikipediaURLCollection:
  def get_random_wikipedia_url(self):
    r = requests.get("https://en.wikipedia.org/wiki/Special:Random", headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True)
    return r.url
  
  def min_word_check(self, url, min_words=200):
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
    soup = BeautifulSoup(html, "html.parser")

    content = soup.find("div", {"id": "mw-content-text"})
    if not content:
        return False

    text = content.get_text(separator=" ")
    return len(text.split()) >= min_words

  def collect_random_urls(self, n):
    random_urls = set()
    while len(random_urls) < n:
        try:
            url = self.get_random_wikipedia_url()
            if url in random_urls and not self.min_word_check(url):
                continue
            random_urls.add(url)
            print(f"Collected {len(random_urls)}/{n}: {url}")
            # time.sleep(1)  
        except Exception as e:
            print("Error:", e)
    return list(random_urls)

  def save_urls_to_json(self, url_list, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(url_list, f, indent=2)


class Preprocessing:
  def extract_wiki_text(self, url):
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
    soup = BeautifulSoup(html, "html.parser")

    # remove unwanted parts
    for tag in soup(["table", "sup", "style", "script"]):
        tag.decompose()

    title = soup.find("h1").get_text()
    content = soup.find("div", {"id": "mw-content-text"})

    text = content.get_text(separator=" ", strip=True)
    return title, text

#   def chunk_text(self, text, min_tokens=200, max_tokens=400, overlap=50):
#     print("Chunking text...")
#     tokens = tokenizer.encode(text)
#     chunks = []
#     start = 0
#     while start < len(tokens):
#         end = min(start + max_tokens, len(tokens))
#         chunk_tokens = tokens[start:end]

#         if len(chunk_tokens) >= min_tokens:
#             chunk_text = tokenizer.decode(chunk_tokens)
#             chunks.append(chunk_text)
#         start = end - overlap
#     print("Chunks created:", chunks)
#     return chunks
  
  def chunk_text(self, text, min_tokens=200, max_tokens=400, overlap=50):
    # print("Chunking text...")
    tokens = tokenizer.encode(text)
    step = max_tokens - overlap
    return [
        tokenizer.decode(tokens[i:min(i + max_tokens, len(tokens))])
        for i in range(0, len(tokens), step)
        if min(i + max_tokens, len(tokens)) - i >= min_tokens
    ]


  def process_url(self, url, source_type):
    title, text = self.extract_wiki_text(url)
    chunks = self.chunk_text(text)
    # print("printing chunks:", chunks)

    records = []
    for i, chunk in enumerate(chunks):
        records.append({
            "chunk_id": str(uuid.uuid4()),
            "url": url,
            "title": title,
            "chunk_index": i,
            "text": chunk,
            "source_type": source_type  # "fixed" or "random"
        })
    # print(f"Processed URL: {url}, Chunks created: {len(chunks)}")
    return records

  def save_chunks(self, all_chunks, filename="wiki_chunks.jsonl"):
    with open(filename, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

  def load_urls_from_json(self, filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)
    
  def load_chunks(self, filename="wiki_chunks.jsonl"):
    chunks = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


class DenseRetriever:
    def __init__(self, chunks, model_name="all-MiniLM-L6-v2"):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]

        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            self.texts,
            normalize_embeddings=True,
            show_progress_bar=True
        ).astype("float32")

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity
        self.index.add(self.embeddings)

    def retrieve(self, query, top_k=5):
        query_emb = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(query_emb, top_k)
        return [self.chunks[i] for i in indices[0]]

class SparseRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.tokenized_corpus = [
            c["text"].lower().split() for c in chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query, top_k=5):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        return [self.chunks[i] for i in top_indices]


class RRF:
    def __init__(self, dense_retriever, sparse_retriever, k=60):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.k = k

    def retrieve(self, query, top_k=5, final_n=5):
        dense_results = self.dense.retrieve(query, top_k)
        sparse_results = self.sparse.retrieve(query, top_k)

        scores = defaultdict(float)

        for rank, doc in enumerate(dense_results):
            scores[doc["chunk_id"]] += 1 / (self.k + rank + 1)

        for rank, doc in enumerate(sparse_results):
            scores[doc["chunk_id"]] += 1 / (self.k + rank + 1)

        # map chunk_id → chunk
        chunk_map = {
            c["chunk_id"]: c for c in (dense_results + sparse_results)
        }

        ranked_chunks = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [chunk_map[cid] for cid, _ in ranked_chunks[:final_n]]

class ResponseGenerator:
    def __init__(self, model_name="distilgpt2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, query, chunks, max_input_tokens=1024, max_output_tokens=150):
        """
        query: str
        chunks: list of chunk dicts [{"text": ...}]
        """
        # 1. Concatenate top-N chunk texts
        context_text = "\n".join([c["text"] for c in chunks])
        prompt = f"""You are an AI assistant. Read the context carefully and provide a detailed answer.
            Use only the information from the context. If the answer is not in the context, say "I don't know."

            Context:
            {context_text}

            Question: {query}

            Answer step by step:"""

        print("Prompt (tokens):", prompt)
        # 2. Tokenize and truncate to model input limit
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens
        ).to(self.device)

        # 3. Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_output_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        # 4. Decode and return answer
        answer = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return answer


if __name__ == "__main__":

    ## Uncomment below to collect URLs
    # wiki_obj = WikipediaURLCollection()
    # fixed_url = wiki_obj.collect_random_urls(200)
    # wiki_obj.save_urls_to_json(fixed_url, "fixed_urls.json")
    # random_url = wiki_obj.collect_random_urls(300)
    # wiki_obj.save_urls_to_json(random_url, "random_urls.json")

    # Preprocessing and chunking
    preprocess = Preprocessing()
    all_chunks = []

    fixed_urls = preprocess.load_urls_from_json("fixed_urls.json")
    random_urls = preprocess.load_urls_from_json("random_urls.json")

    for url in fixed_urls:
        all_chunks.extend(preprocess.process_url(url, "fixed"))

    for url in random_urls:
        all_chunks.extend(preprocess.process_url(url, "random"))

    preprocess.save_chunks(all_chunks, "wiki_chunks.jsonl")
    print(f"Total chunks created: {len(all_chunks)}")


    chunks = preprocess.load_chunks("wiki_chunks.jsonl")

    dense = DenseRetriever(chunks)
    sparse = SparseRetriever(chunks)
    rrf = RRF(dense, sparse)

    # query = "Give list of ambassadors of Germany to Netherlands"

    # results = rrf.retrieve(query, top_k=10, final_n=1)

    # for r in results:
    #     print(r["title"])
    #     print(r["text"][:150])
    #     print("-" * 40)

    generator = ResponseGenerator()
    # answer = generator.generate(query, results)
    # print("answer length:", len(answer))

    # print("results:", results)
    # print("\n\n\n==============Answer:\n", answer)

    st.set_page_config(page_title="RAG QA System", layout="wide")

    st.title("📚 RAG QA System")

    # User input
    query = st.text_input("Enter your question:")

    # Optional: top-N chunks slider
    top_n = st.slider("Number of top chunks to use", 1, 10, 5)

    if query:
        start_time = time.time()
        
        # ----------------------------
        # 1️⃣ Retrieve top chunks
        # ----------------------------
        # Example: replace with your RRF retriever
        top_chunks = rrf.retrieve(query, top_k=10, final_n=top_n)

        # Convert chunks to dataframe for display
        chunks_df = pd.DataFrame([
            {
                "Chunk Index": c["chunk_index"],
                "URL": c["url"],
                "Text": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
                "Dense Score": c.get("dense_score", ""),
                "Sparse Score": c.get("sparse_score", ""),
                "RRF Score": c.get("rrf_score", "")
            }
            for c in top_chunks
        ])
        
        # ----------------------------
        # 2️⃣ Generate answer
        # ----------------------------
        answer = generator.generate(query, top_chunks)
        
        end_time = time.time()
        elapsed = end_time - start_time

        # ----------------------------
        # 3️⃣ Display results
        # ----------------------------
        st.subheader("📝 Generated Answer")
        st.write(answer)

        st.subheader("📄 Top Retrieved Chunks")
        st.dataframe(chunks_df)

        st.write(f"⏱ Response time: {elapsed:.2f} seconds")
