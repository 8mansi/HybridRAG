# pip install sentence_transformers
# !pip install requests beautifulsoup4 
# !pip install tiktoken
# pip install faiss-cpu
# pip install rank-bm25
# pip install streamlit
# pip install transformers torch
# pip install pandas
# python -m streamlit run .\HybridRag.py

import re
import os
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
from concurrent.futures import ThreadPoolExecutor
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
import unicodedata


tokenizer = tiktoken.get_encoding("cl100k_base")

from concurrent.futures import ThreadPoolExecutor, as_completed

class WikipediaURLCollection:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def get_random_wikipedia_url(self, min_words=200):
        try:
            r = self.session.get(
                "https://en.wikipedia.org/wiki/Special:RandomInCategory/Physics",
                allow_redirects=True,
                timeout=10
            )
            url = r.url

            soup = BeautifulSoup(r.text, "html.parser")
            content = soup.find("div", {"id": "mw-content-text"})
            if not content:
                print("No content found for URL:", url)
                return None

            text = content.get_text(separator=" ")
            if len(text.split()) < min_words:
                print(f"URL {url} has less than {min_words} words.")
                return None

            return url
        except Exception as e:
            print("Error fetching random Wikipedia URL, e = ", e)
            return None
        
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
                # if not self.min_word_check(url):
                #     continue
                if url is None:
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

        title = soup.find("h1").get_text()
        content = soup.find("div", {"id": "mw-content-text"})

        content = self.clean_wikipedia_html(content)

        text = content.get_text(separator=" ", strip=True)
        text = self.post_clean_text(text)
        return title, text
    
    def post_clean_text(self, text):
        text = re.sub(r"\[.*?\]", "", text)  # remove [1], [edit], [citation needed]
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def clean_wikipedia_html(self, soup):
        # Remove tables (infobox, navbox, etc.)
        for tag in soup.find_all("table"):
            tag.decompose()

        # Remove citation superscripts [1], [2], etc.
        for sup in soup.find_all("sup", class_="reference"):
            sup.decompose()

        # Remove reference lists
        for div in soup.find_all("div", class_=["reflist", "refbegin"]):
            div.decompose()

        for ol in soup.find_all("ol", class_="references"):
            ol.decompose()

        # Remove edit section links
        for span in soup.find_all("span", class_="mw-editsection"):
            span.decompose()

        # Remove navigation boxes
        for div in soup.find_all("div", class_=["navbox", "vertical-navbox"]):
            div.decompose()

        # Remove footnotes
        for div in soup.find_all("div", role="note"):
            div.decompose()

        # Remove scripts and styles
        for tag in soup(["script", "style"]):
            tag.decompose()

        return soup


    def clean_text(self,text: str) -> str:
        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+", " ", text)

        # Remove emails
        text = re.sub(r"\S+@\S+", " ", text)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Remove control characters
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)

        # Keep ONLY English letters, numbers, and sentence punctuation
        text = re.sub(r"[^a-z0-9\s\.\,\?\!\-']", " ", text)

        # Remove single-character noise (keep a, i)
        text = re.sub(r"\b(?!a\b|i\b)[a-z]\b", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    
    def chunk_text(self, text, min_tokens=200, max_tokens=400, overlap=50):
        # print("Chunking text...")
        clean_text = self.clean_text(text)
        print("raw text: ",text[:500],"\n clean text: ",clean_text[:500])
        tokens = tokenizer.encode(clean_text)
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

        if os.path.exists("dense.index") and os.path.exists("embeddings.npy"):
            self.index = faiss.read_index("dense.index")
            self.embeddings = np.load("embeddings.npy")
            return
        
        self.embeddings = self.model.encode(
            self.texts,
            normalize_embeddings=True,
            show_progress_bar=True
        ).astype("float32")

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity
        self.index.add(self.embeddings)
        faiss.write_index(self.index, "dense.index")
        np.save("embeddings.npy", self.embeddings)

    def retrieve(self, query, top_k=5):
        query_emb = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(query_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            chunk = self.chunks[idx].copy()
            chunk["dense_score"] = float(score)  # store score
            results.append(chunk)
        return results

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

        results = []
        for i in top_indices:
            chunk = self.chunks[i].copy()
            chunk["sparse_score"] = float(scores[i])  # store score
            results.append(chunk)
        return results
    

class RRF:
    def __init__(self, dense_retriever, sparse_retriever, k=60):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.k = k

    def retrieve(self, query, top_k=5, final_n=5):
        dense_results = self.dense.retrieve(query, top_k)
        sparse_results = self.sparse.retrieve(query, top_k)

        rrf_scores = defaultdict(float)
        for rank, chunk in enumerate(dense_results):
            rrf_scores[chunk["chunk_id"]] += 1 / (self.k + rank + 1)
        for rank, chunk in enumerate(sparse_results):
            rrf_scores[chunk["chunk_id"]] += 1 / (self.k + rank + 1)

        chunk_map = {}
        for chunk in dense_results + sparse_results:
            cid = chunk["chunk_id"]
            if cid not in chunk_map:  # first occurrence
                chunk_copy = chunk.copy()
                chunk_copy["rrf_score"] = float(rrf_scores[cid])
                chunk_map[cid] = chunk_copy
            else:  # update rrf_score if needed
                chunk_map[cid]["rrf_score"] = float(rrf_scores[cid])

        ranked_chunks = sorted(
            chunk_map.values(),
            key=lambda x: (
                x.get("rrf_score", 0.0),
                x.get("sparse_score", 0.0)
            ),
            reverse=True
        )

        return ranked_chunks[:final_n]

class ResponseGenerator:
    def __init__(self, model_name="google/flan-t5-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()
    def summarize_chunks(self, texts, max_tokens=120):
        prompts = [
            f"Summarize briefly:\n{text}" for text in texts
        ]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False
            )

        return [
            self.tokenizer.decode(o, skip_special_tokens=True)
            for o in outputs
        ]

    def generate(self, query, chunks, max_input_tokens=512, max_output_tokens=150):
        summaries = self.summarize_chunks([c["text"] for c in chunks])

        context_text = "\n\n".join(summaries)

        prompt = f"""
        Context:
        {context_text}

        Question:
        {query}

        Answer in one or two sentences based only on the context.
        """

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_output_tokens,
                do_sample=False
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


@st.cache_resource
def load_dense_retriever(chunks):
    return DenseRetriever(chunks)

@st.cache_resource
def load_sparse_retriever(chunks):
    return SparseRetriever(chunks)

@st.cache_resource
def load_generator():
    return ResponseGenerator()

backend_ready = False
dense = None
sparse = None
rrf = None
generator = None

def setup_backend():
    print("Setting up backend...")
    global dense, sparse, rrf, generator, backend_ready

    wiki_obj = WikipediaURLCollection()

    if not os.path.exists("fixed_urls.json"):
        fixed_url = wiki_obj.collect_random_urls(200)
        print("Fixed URLs collected.")
        wiki_obj.save_urls_to_json(fixed_url, "fixed_urls.json")

    if not os.path.exists("random_urls.json"):
        random_url = wiki_obj.collect_random_urls(300)
        print("Random URLs collected.")
        wiki_obj.save_urls_to_json(random_url, "random_urls.json")

    preprocess = Preprocessing()
    if not os.path.exists("wiki_chunks.jsonl"):
        all_chunks = []

        fixed_urls = preprocess.load_urls_from_json("fixed_urls.json")
        random_urls = preprocess.load_urls_from_json("random_urls.json")

        def process_url_safe(url, source_type):
            try:
                return preprocess.process_url(url, source_type)
            except Exception as e:
                print(f"Error processing {url}: {e}")
                return []
        urls_to_process = [(url, "fixed") for url in fixed_urls] + [(url, "random") for url in random_urls]
        with ThreadPoolExecutor(max_workers=8) as executor:
            for chunks in executor.map(lambda p: process_url_safe(*p), urls_to_process):
                all_chunks.extend(chunks)

        preprocess.save_chunks(all_chunks, "wiki_chunks.jsonl")
        chunks = all_chunks
    else:
        chunks = preprocess.load_chunks("wiki_chunks.jsonl")
    dense = load_dense_retriever(chunks)
    sparse = load_sparse_retriever(chunks)
    rrf = RRF(dense, sparse)
    generator = load_generator()

    st.session_state.backend_ready = True
    print("Backend is ready.")

setup_backend()

st.set_page_config(page_title="RAG QA System", layout="wide")
st.title("RAG QA System")

query = st.text_input("Enter your question:")
top_n = st.slider("Number of top chunks to use", 1, 10, 5)

if st.session_state.backend_ready:
    if st.button("Get Answer") and query.strip():
        print("Retrieving and generating answer...")
        start_time = time.time()
        top_chunks = rrf.retrieve(query, top_k=10, final_n=top_n)
        chunks_df = pd.DataFrame([
            {
                "Chunk Index": c["chunk_index"],
                "URL": c["url"],
                "Text": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
                "Dense Score": c.get("dense_score", 0),
                "Sparse Score": c.get("sparse_score", 0),
                "RRF Score": c.get("rrf_score", 0)
            }
            for c in top_chunks
        ])
        answer = generator.generate(query, top_chunks)
        elapsed = time.time() - start_time

        st.subheader("Generated Answer")
        st.write(answer)

        st.subheader("Top Retrieved Chunks")
        st.dataframe(chunks_df)

        st.write(f"Response time: {elapsed:.2f} seconds")
