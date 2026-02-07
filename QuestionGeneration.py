import json
import random
import torch
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

class QAGenerator:
    """
    Question-Answer Generation System
    
    Generates Q&A pairs from Wikipedia chunks using a language model.
    Supports multiple question types (factual, comparative, inferential, multi-hop).
    """
    
    QUESTION_TYPES = ["factual", "comparative", "inferential", "multi_hop"]
    DEFAULT_TOTAL_QA = 10
    DEFAULT_CHUNK_SAMPLE_SIZE = 180
    
    def __init__(self, generator=None, total_qa=DEFAULT_TOTAL_QA, 
                 chunk_sample_size=DEFAULT_CHUNK_SAMPLE_SIZE):
        """
        Initialize QA Generator
        
        Args:
            generator: Language model generator (from HybridRag.load_generator())
            total_qa: Number of Q&A pairs to generate
            chunk_sample_size: Number of chunks to sample from
        """
        if generator is None:
            # Lazy import to avoid circular dependency
            from HybridRag import load_generator
            generator = load_generator()
        self.generator = generator
        self.total_qa = total_qa
        self.chunk_sample_size = chunk_sample_size
        self.dataset = []
    
    @staticmethod
    def load_chunks(path="wiki_chunks.jsonl"):
        """Load all chunks from JSONL file"""
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    
    @staticmethod
    def parse_qa(text):
        """Parse Q&A from raw text"""
        text = text.strip()
        if "Question:" in text and "Answer:" in text:
            q, a = text.split("Answer:", 1)
            question = q.replace("Question:", "").strip()
            answer = a.strip()
            return question, answer
        if text.endswith("?"):
            return text, "Answer not explicitly generated"
        return None, None
    
    def generate_question(self, chunk, qtype):
        """Generate a question from a chunk and question type"""
        q_prompt = (
            f"Context:\n{chunk['text']}\n\n"
            f"Generate ONE {qtype} question.\n"
            f"Rules:\n"
            f"- Must be answerable from the context\n"
            f"- End with '?'\n"
            f"- One sentence only\n"
            f"Question:"
        )

        q_inputs = self.generator.tokenizer(
            q_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.generator.device)

        with torch.no_grad():
            q_out = self.generator.model.generate(
                **q_inputs,
                max_new_tokens=48,
                do_sample=False
            )

        decoded_q = self.generator.tokenizer.decode(
            q_out[0],
            skip_special_tokens=True
        )

        # Extract question safely
        question = decoded_q.split("Question:")[-1].strip()
        if "?" not in question:
            return None
        return question.split("?")[0].strip() + "?"
    
    def generate_answer(self, chunk, question):
        """Generate an answer for a question from a chunk"""
        a_prompt = (
            f"Context:\n{chunk['text']}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer in ONE short phrase or sentence.\n"
            f"Answer:"
        )

        a_inputs = self.generator.tokenizer(
            a_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.generator.device)

        with torch.no_grad():
            a_out = self.generator.model.generate(
                **a_inputs,
                max_new_tokens=32,
                do_sample=False
            )

        decoded_a = self.generator.tokenizer.decode(
            a_out[0],
            skip_special_tokens=True
        )

        return decoded_a.split("Answer:")[-1].strip()
    
    def validate_answer(self, answer):
        """Validate generated answer"""
        if (
            len(answer) < 2
            or len(answer.split()) > 20
            or "context" in answer.lower()
            or "question" in answer.lower()
        ):
            return False
        return True
    
    def generate_qa(self, chunk, qtype):
        """
        Generate a Q&A pair from a chunk
        
        Args:
            chunk: Document chunk with 'text', 'chunk_id', 'url', 'title'
            qtype: Question type (factual, comparative, inferential, multi_hop)
        
        Returns:
            dict: Q&A pair with metadata, or None if generation failed
        """
        # Generate question
        question = self.generate_question(chunk, qtype)
        if question is None:
            return None
        
        # Generate answer
        answer = self.generate_answer(chunk, question)
        
        # Validate answer
        if not self.validate_answer(answer):
            return None
        
        return {
            "question": question,
            "answer": answer,
            "question_type": qtype,
            "source_id": chunk["chunk_id"],
            "source_url": chunk["url"],
            "title": chunk["title"]
        }
    
    def generate_dataset(self, chunks):
        """
        Generate a dataset of Q&A pairs from chunks
        
        Args:
            chunks: List of document chunks
        
        Returns:
            list: Generated Q&A pairs
        """
        self.dataset = []
        
        # Sample chunks
        sampled_chunks = random.sample(
            chunks,
            min(self.chunk_sample_size, len(chunks))
        )
        
        # Pre-assign question types evenly
        qtypes = (
            self.QUESTION_TYPES * (self.total_qa // len(self.QUESTION_TYPES))
            + random.choices(self.QUESTION_TYPES, k=self.total_qa % len(self.QUESTION_TYPES))
        )
        random.shuffle(qtypes)
        
        # Generate Q&A pairs
        for chunk, qtype in tqdm(zip(sampled_chunks, qtypes), total=self.total_qa, 
                                  desc="Generating Q&A pairs"):
            if len(self.dataset) >= self.total_qa:
                break
            
            qa = self.generate_qa(chunk, qtype)
            if qa:
                self.dataset.append(qa)
        
        return self.dataset
    
    def save_dataset(self, json_file="qa_dataset.json", csv_file="qa_dataset.csv"):
        """
        Save generated dataset to JSON and CSV
        
        Args:
            json_file: Path to save JSON file
            csv_file: Path to save CSV file
        """
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, indent=2)
        pd.DataFrame(self.dataset).to_csv(csv_file, index=False)
        print(f"Dataset saved: {json_file}, {csv_file}")


class MRREvaluator:
    """
    Mean Reciprocal Rank (MRR) Evaluator at URL Level
    
    Calculates MRR metric for Hybrid RAG system at the URL level.
    For each question, finds the rank position of the first correct Wikipedia URL 
    in retrieved results. MRR = average of 1/rank across all questions.
    """
    
    def __init__(self, qa_dataset_path="qa_dataset.json", chunks_path="wiki_chunks.jsonl"):
        """Initialize evaluator with QA dataset and chunks"""
        self.qa_dataset = self._load_qa_dataset(qa_dataset_path)
        self.chunks = self._load_chunks(chunks_path)
        self.results = None
    
    @staticmethod
    def _load_qa_dataset(path="qa_dataset.json"):
        """Load QA dataset"""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    @staticmethod
    def _load_chunks(path="wiki_chunks.jsonl"):
        """Load all chunks"""
        chunks = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        return chunks
    
    def evaluate(self, retriever, top_k=10, system_name="Hybrid RAG"):
        """
        Evaluate MRR for a given retriever
        
        Args:
            retriever: Object with retrieve(query, top_k) method that returns list of chunks
            top_k: Number of top results to consider
            system_name: Name of the retrieval system
        
        Returns:
            dict: Evaluation results including MRR, details, and statistics
        """
        
        reciprocal_ranks = []
        ranks = []
        found_count = 0
        details = []
        
        for idx, qa in enumerate(tqdm(self.qa_dataset, desc=f"Evaluating {system_name}")):
            question = qa["question"]
            correct_url = qa["source_url"]
            
            # Retrieve results
            retrieved = retriever.retrieve(question, top_k=top_k)
            
            # Find rank of first correct URL
            rank = None
            for position, chunk in enumerate(retrieved, start=1):
                if chunk["url"] == correct_url:
                    rank = position
                    break
            
            # Calculate reciprocal rank
            if rank is not None:
                reciprocal_rank = 1.0 / rank
                reciprocal_ranks.append(reciprocal_rank)
                ranks.append(rank)
                found_count += 1
            else:
                reciprocal_ranks.append(0.0)
                ranks.append(0)
            
            # Store details
            details.append({
                "question_id": idx,
                "question": question,
                "correct_url": correct_url,
                "expected_answer": qa.get("answer", ""),
                "rank": rank,
                "reciprocal_rank": reciprocal_ranks[-1],
                "question_type": qa.get("question_type", "unknown"),
                "retrieved_urls": [ch["url"] for ch in retrieved]
            })
        
        # Calculate MRR
        mrr = sum(reciprocal_ranks) / len(self.qa_dataset) if self.qa_dataset else 0.0
        
        result = {
            "system": system_name,
            "mrr": mrr,
            "reciprocal_ranks": reciprocal_ranks,
            "ranks": ranks,
            "found_count": found_count,
            "total_count": len(self.qa_dataset),
            "coverage": found_count / len(self.qa_dataset) if self.qa_dataset else 0.0,
            "top_k": top_k,
            "details": details
        }
        # Compute additional retrieval metrics (Precision@K and HitRate@K)
        ks = [1, 3, 5, 10]
        metrics = {}
        for k in ks:
            metrics[f"precision@{k}"] = self.precision_at_k(result, k)
            metrics[f"hit_rate@{k}"] = self.hit_rate_at_k(result, k)
        result["metrics"] = metrics

        self.results = result
        return result

    def precision_at_k(self, result, k=10):
        """Compute Precision@K for the evaluation result.

        Precision@K (per-query) = (# of relevant documents in top-K) / K.
        With a single relevant URL per question this is either 1/K or 0.
        We return the mean Precision@K across all queries.
        """
        if result is None or "details" not in result or result["total_count"] == 0:
            return 0.0
        precisions = []
        for det in result["details"]:
            topk = det.get("retrieved_urls", [])[:k]
            rel_count = 1 if det.get("correct_url") in topk else 0
            precisions.append(rel_count / float(k))
        return sum(precisions) / len(precisions)

    def hit_rate_at_k(self, result, k=10):
        """Compute HitRate@K (also Recall@K for single relevant doc).

        HitRate@K = fraction of queries where the correct URL appears in top-K.
        This is often called Recall@K or Hit@K when there's one relevant item.
        """
        if result is None or "details" not in result or result["total_count"] == 0:
            return 0.0
        hits = 0
        for det in result["details"]:
            topk = det.get("retrieved_urls", [])[:k]
            if det.get("correct_url") in topk:
                hits += 1
        return hits / float(result["total_count"]) if result["total_count"] else 0.0

if __name__ == "__main__":
    qa_gen = QAGenerator(total_qa=10, chunk_sample_size=180)
    chunks = QAGenerator.load_chunks("wiki_chunks.jsonl")
    dataset = qa_gen.generate_dataset(chunks)
    qa_gen.save_dataset("qa_dataset.json", "qa_dataset.csv")
    
    print(f"\nGenerated {len(dataset)} Q&A pairs")
    
    try:
        from HybridRag import DenseRetriever, SparseRetriever, RRF
        evaluator = MRREvaluator("qa_dataset.json", "wiki_chunks.jsonl")
        
        # Initialize retrievers
        dense_retriever = DenseRetriever(chunks)
        sparse_retriever = SparseRetriever(chunks)
        rrf_retriever = RRF(dense_retriever, sparse_retriever)
        
        result_rrf = evaluator.evaluate(rrf_retriever, top_k=10, system_name="Hybrid RAG (RRF)")
        result_dense = evaluator.evaluate(dense_retriever, top_k=10, system_name="Dense Retrieval")
        result_sparse = evaluator.evaluate(sparse_retriever, top_k=10, system_name="Sparse Retrieval (BM25)")

    except ImportError as e:
        print(f"\nSkipping MRR evaluation: {e}")
        print("Make sure HybridRag.py is available with required retrievers.")