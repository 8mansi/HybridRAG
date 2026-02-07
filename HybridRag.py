# pip install sentence_transformers
# !pip install requests beautifulsoup4 
# !pip install tiktoken
# pip install faiss-cpu
# pip install rank-bm25
# pip install streamlit
# pip install transformers torch
# pip install pandas
# python -m streamlit run .\HybridRag.py

import os
import pandas as pd
import requests, json, time, random
from bs4 import BeautifulSoup
import tiktoken
import uuid
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from collections import defaultdict
import torch
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
import unicodedata
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from QuestionGeneration import QAGenerator, MRREvaluator
import matplotlib.pyplot as plt

tokenizer = tiktoken.get_encoding("cl100k_base")

class WikipediaURLCollection:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def get_pages_from_category(self, category, max_pages=1000, depth=2):
        visited_categories = set()
        collected_pages = set()

        def crawl(cat, current_depth):
            if current_depth > depth:
                return
            if cat in visited_categories:
                return
            if len(collected_pages) >= max_pages:
                return

            visited_categories.add(cat)

            url = "https://en.wikipedia.org/w/api.php"
            cmcontinue = None

            while True:
                params = {
                    "action": "query",
                    "list": "categorymembers",
                    "cmtitle": f"Category:{cat}",
                    "cmlimit": "500",
                    "format": "json",
                }

                if cmcontinue:
                    params["cmcontinue"] = cmcontinue

                try:
                    r = self.session.get(
                        url,
                        params=params,
                        headers={
                            "User-Agent": "HybridRAG-ResearchBot/1.0 (MTech Conversational AI Project)"
                        },
                        timeout=10,
                    )

                    if r.status_code != 200:
                        print("Bad status code:", r.status_code)
                        return

                    if not r.text.strip():
                        print("Empty response received")
                        return

                    data = r.json()

                except Exception as e:
                    print("Request failed:", e)
                    return

                if "query" not in data:
                    return

                for member in data["query"]["categorymembers"]:
                    if member["ns"] == 0:  # Article
                        page_url = (
                            "https://en.wikipedia.org/wiki/"
                            + member["title"].replace(" ", "_")
                        )
                        collected_pages.add(page_url)

                    elif member["ns"] == 14:  # Subcategory
                        subcat = member["title"].replace("Category:", "")
                        crawl(subcat, current_depth + 1)

                    if len(collected_pages) >= max_pages:
                        return

                if "continue" in data:
                    cmcontinue = data["continue"]["cmcontinue"]
                else:
                    break

                time.sleep(0.5)  # ← prevent Wikipedia rate limit

        crawl(category, 0)

        return list(collected_pages)[:max_pages]


    def collect_random_urls(self, n):
        all_pages = self.get_pages_from_category("Physics", max_pages=n * 3, depth=1)
        random.shuffle(all_pages)
        return all_pages[:n]

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
        if soup is None:
            return BeautifulSoup("", "html.parser")
        
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
        clean_text = self.clean_text(text)
        # print("raw text: ",text[:500],"\n clean text: ",clean_text[:500])
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
    def __init__(self, chunks, model_name="all-mpnet-base-v2"):
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
        clean_query = query.lower()
        query_emb = self.model.encode([clean_query], normalize_embeddings=True)

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
        print("Initializing RRF...")
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
    def __init__(self, model_name="google/flan-t5-large", device=None):
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
        context_text = "\n\n".join([c["text"] for c in chunks])

        prompt = f"""
            Write a detailed explanation in paragraph form based only on the context.

            Context:
            ----------------
            {context_text}
            ----------------

            Question:
            {query}

            Answer:
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

def setup_backend():
    if st.session_state.get("backend_ready", False):
        return

    print("Setting up backend...")

    wiki_obj = WikipediaURLCollection()

    if not os.path.exists("fixed_urls.json"):
        fixed_url = wiki_obj.collect_random_urls(200)
        print("Fixed URLs collected.")
        wiki_obj.save_urls_to_json(fixed_url, "fixed_urls.json")

    # if not os.path.exists("random_urls.json"):
    random_url = wiki_obj.collect_random_urls(300)
    print("Random URLs collected.")
    wiki_obj.save_urls_to_json(random_url, "random_urls.json")

    preprocess = Preprocessing()
    # if not os.path.exists("wiki_chunks.jsonl"):
    all_chunks = []

    fixed_urls = preprocess.load_urls_from_json("fixed_urls.json")
    random_urls = preprocess.load_urls_from_json("random_urls.json")
    all_urls = list(set(fixed_urls + random_urls))
    def process_url_safe(url, source_type):
        try:
            return preprocess.process_url(url, source_type)
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return []
    urls_to_process = [(url, "mixed") for url in all_urls]
    with ThreadPoolExecutor(max_workers=8) as executor:
        for chunks in executor.map(lambda p: process_url_safe(*p), urls_to_process):
            all_chunks.extend(chunks)

    preprocess.save_chunks(all_chunks, "wiki_chunks.jsonl")
    chunks = all_chunks
    # else:
    #     chunks = preprocess.load_chunks("wiki_chunks.jsonl")
    st.session_state.dense = load_dense_retriever(chunks)
    st.session_state.sparse = load_sparse_retriever(chunks)
    st.session_state.rrf = RRF(
        st.session_state.dense,
        st.session_state.sparse
    )
    st.session_state.generator = load_generator()

    st.session_state.backend_ready = True    
    print("Backend is ready.")


st.set_page_config(page_title="RAG QA System", layout="wide")
st.title("RAG QA System")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Question Answering", "QA Dataset & MRR Evaluation"])

if "backend_initialized" not in st.session_state:
    setup_backend()
    st.session_state.backend_initialized = True

if st.session_state.backend_ready and "qa_mrr_generated" not in st.session_state:
    try:        
        if 'QuestionGeneration' in sys.modules:
            del sys.modules['QuestionGeneration']
        
        # Load chunks
        preprocess_inst = Preprocessing()
        chunks = preprocess_inst.load_chunks("wiki_chunks.jsonl")
        
        # Generate Q&A dataset
        qa_gen = QAGenerator(
            generator=st.session_state.generator,
            total_qa=10,
            chunk_sample_size=300
        )
        dataset = qa_gen.generate_dataset(chunks)
        qa_gen.save_dataset("qa_dataset.json", "qa_dataset.csv")
        
        # Evaluate with MRR
        evaluator = MRREvaluator("qa_dataset.json", "wiki_chunks.jsonl")
        
        # Evaluate all three systems
        result_rrf = evaluator.evaluate(
            st.session_state.rrf, 
            top_k=10, 
            system_name="Hybrid RAG (RRF)"
        )
        
        result_dense = evaluator.evaluate(
            st.session_state.dense, 
            top_k=10, 
            system_name="Dense Retrieval"
        )
        
        result_sparse = evaluator.evaluate(
            st.session_state.sparse, 
            top_k=10, 
            system_name="Sparse Retrieval (BM25)"
        )

        # Store in session state
        st.session_state.result_rrf = result_rrf
        st.session_state.result_dense = result_dense
        st.session_state.result_sparse = result_sparse
        st.session_state.qa_mrr_generated = True
        st.rerun()
        
    except Exception as e:
        print(f"Error during auto-generation: {str(e)}")
        traceback.print_exc()
        st.session_state.qa_mrr_error = str(e)

with tab1:
    query = st.text_input("Enter your question:", key="qa_query_input")
    top_n = st.slider("Number of top chunks to use", 1, 10, 5, key="qa_top_n_slider")

    if st.session_state.backend_ready:
        if st.button("Get Answer", key="get_answer_btn") and query.strip():
            print("Retrieving and generating answer...")
            start_time = time.time()
            top_chunks = st.session_state.rrf.retrieve(query, top_k=10, final_n=top_n)
            print(f"Top chunks retrieved: {len(top_chunks)}")

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
            answer = st.session_state.generator.generate(query, top_chunks)
            elapsed = time.time() - start_time

            st.subheader("Generated Answer")
            st.write(answer)

            st.subheader("Top Retrieved Chunks")
            st.dataframe(chunks_df)

            st.write(f"Response time: {elapsed:.2f} seconds")

with tab2:
    st.subheader("QA Dataset & MRR Evaluation Report")
    
    if "qa_mrr_generated" in st.session_state and st.session_state.qa_mrr_generated:
        result_rrf = st.session_state.result_rrf
        result_dense = st.session_state.result_dense
        result_sparse = st.session_state.result_sparse
        
        comparison_df = pd.DataFrame([
            {
                "System": "Hybrid RAG (RRF)",
                "MRR": f"{result_rrf['mrr']:.4f}",
                "Coverage %": f"{result_rrf['coverage']:.2%}",
                "Precision@3": f"{result_rrf['metrics'].get('precision@3', 0.0):.4f}",
                "HitRate@3": f"{result_rrf['metrics'].get('hit_rate@3', 0.0):.4f}",
                "Top-3": sum(1 for r in result_rrf['ranks'] if 0 < r <= 3),
                "Top-5": sum(1 for r in result_rrf['ranks'] if 0 < r <= 5),
                "Top-10": sum(1 for r in result_rrf['ranks'] if 0 < r <= 10),
            },
            {
                "System": "Dense Retrieval",
                "MRR": f"{result_dense['mrr']:.4f}",
                "Coverage %": f"{result_dense['coverage']:.2%}",
                "Precision@3": f"{result_dense['metrics'].get('precision@3', 0.0):.4f}",
                "HitRate@3": f"{result_dense['metrics'].get('hit_rate@3', 0.0):.4f}",
                "Top-3": sum(1 for r in result_dense['ranks'] if 0 < r <= 3),
                "Top-5": sum(1 for r in result_dense['ranks'] if 0 < r <= 5),
                "Top-10": sum(1 for r in result_dense['ranks'] if 0 < r <= 10),
            },
            {
                "System": "Sparse (BM25)",
                "MRR": f"{result_sparse['mrr']:.4f}",
                "Coverage %": f"{result_sparse['coverage']:.2%}",
                "Precision@3": f"{result_sparse['metrics'].get('precision@3', 0.0):.4f}",
                "HitRate@3": f"{result_sparse['metrics'].get('hit_rate@3', 0.0):.4f}",
                "Top-3": sum(1 for r in result_sparse['ranks'] if 0 < r <= 3),
                "Top-5": sum(1 for r in result_sparse['ranks'] if 0 < r <= 5),
                "Top-10": sum(1 for r in result_sparse['ranks'] if 0 < r <= 10),
            }
        ])
        st.dataframe(comparison_df, width='stretch')
        st.subheader(" Questions & Retrieval Reports")
        
        if result_rrf['details']:
            num_questions = len(result_rrf['details'])
            display_limit = min(20, num_questions)
            
            for idx in range(display_limit):
                detail_rrf = result_rrf['details'][idx]
                detail_dense = result_dense['details'][idx]
                detail_sparse = result_sparse['details'][idx]
                
                question = detail_rrf['question']
                q_type = detail_rrf['question_type']
                correct_url = detail_rrf['correct_url']
                with st.expander(f"Q{idx+1} [{q_type}] {question}", expanded=False):
                    st.write(f"**Correct URL:** {correct_url}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        rank_rrf = detail_rrf['rank']
                        st.write("**RRF**") 
                        st.write("Rank - ", rank_rrf if rank_rrf else "0")
                        st.write(f"MRR - {detail_rrf['reciprocal_rank']:.4f}")
                        st.write("Top 5 Retrieved URLs:")
                        for i, url in enumerate(detail_rrf['retrieved_urls'][:5], 1):
                            status = "✓" if url == correct_url else "✗"
                            st.caption(f"{status} {i}. {url[:90]}")
                    with col2:
                        rank_dense = detail_dense['rank']
                        st.write("**Dense**") 
                        st.write("Rank - ", rank_dense if rank_dense else "0")
                        st.write(f"MRR - {detail_dense['reciprocal_rank']:.4f}")

                        st.write("Top 5 Retrieved URLs:")
                        for i, url in enumerate(detail_dense['retrieved_urls'][:5], 1):
                            status = "✓" if url == correct_url else "✗"
                            st.caption(f"{status} {i}. {url[:90]}")
                    with col3:
                        rank_sparse = detail_sparse['rank']
                        st.write("**Sparse (BM25)**") 
                        st.write("Rank -", rank_sparse if rank_sparse else "0")
                        st.write(f"MRR - {detail_sparse['reciprocal_rank']:.4f}")
                        st.write("Top 5 Retrieved URLs:")
                        for i, url in enumerate(detail_sparse['retrieved_urls'][:5], 1):
                            status = "✓" if url == correct_url else "✗"
                            st.caption(f"{status} {i}. {url[:90]}")

                    ret_col1, ret_col2, ret_col3 = st.columns(3)
                
    elif "qa_mrr_error" in st.session_state:
        st.error(f" Error: {st.session_state.qa_mrr_error}")
    else:
        st.info("Generating QA dataset and MRR report")
