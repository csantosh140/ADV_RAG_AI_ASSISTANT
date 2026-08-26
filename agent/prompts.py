"""Structured prompts for Agentic RAG workflow nodes and evaluation."""

QUERY_REWRITE_PROMPT = """You are an expert search query refiner.
Your task is to take a user's question and reformulate it into 2 optimized search queries for a document retrieval engine.
Focus on extracting key technical entities, keywords, and synonyms.

User Query: {query}

Provide exactly 2 search queries, one per line:"""

QUERY_REWRITE_WITH_HISTORY_PROMPT = """You are an expert conversational retrieval optimizer.
Given the chat history and the user's latest follow-up question, rewrite the question into 2 standalone search queries that incorporate relevant context (resolving pronouns like "it", "they", "that", or references to previous concepts).

CHAT HISTORY:
{chat_history}

LATEST QUESTION:
{query}

Provide exactly 2 standalone search queries, one per line:"""

DOCUMENT_GRADER_PROMPT = """You are an AI grader assessing the relevance of a retrieved document snippet to a user query.

User Query: {query}
Document Snippet:
\"\"\"{document_text}\"\"\"

Does this document snippet contain relevant information or context to answer or partially answer the query?
Answer ONLY with 'YES' or 'NO'."""

GROUNDED_GENERATION_PROMPT = """You are a strictly grounded, enterprise-grade AI assistant.
Your mission is to provide an accurate, clear, and comprehensive answer to the user's question based EXCLUSIVELY on the provided source context chunks.

CRITICAL GROUNDING RULES:
1. Only state facts directly mentioned in the Context. Never extrapolate, speculate, or bring in external knowledge.
2. If the context does not contain sufficient information to answer the question, state: "I do not have sufficient information in the provided documents to answer this question."
3. Include inline numerical citation markers like [1], [2] at the end of each sentence or claim that references a source chunk.
4. Keep the tone professional, objective, and well-structured.

CONTEXT:
{context_blocks}

USER QUESTION:
{query}

GROUNDED ANSWER (with [X] citations):"""

HALLUCINATION_CHECK_PROMPT = """You are a strict groundedness and hallucination evaluation judge.
Your job is to verify whether the AI's generated answer is 100% supported by the given source context.

SOURCE CONTEXT:
{context_blocks}

AI ANSWER:
{answer}

Evaluate if every factual statement in the AI Answer is directly supported by the Source Context.
Output your verdict in the following exact format:
SCORE: <float between 0.0 and 1.0>
GROUNDED: <YES or NO>
REASON: <Brief one sentence explanation>"""

ANSWER_RELEVANCE_EVAL_PROMPT = """You are an AI judge evaluating answer relevance.
Given a user query and an AI-generated answer, evaluate if the answer directly and completely answers the user's query.

USER QUERY: {query}
AI ANSWER: {answer}

Output your verdict in the following exact format:
SCORE: <float between 0.0 and 1.0>
REASON: <Brief one sentence explanation>"""

CONTEXT_PRECISION_EVAL_PROMPT = """You are an AI judge evaluating context retrieval quality against a reference answer.

REFERENCE ANSWER / GROUND TRUTH:
{ground_truth}

RETRIEVED CONTEXT:
{context_blocks}

Does the retrieved context contain the key information needed to answer the reference question?
Output your verdict in the following exact format:
SCORE: <float between 0.0 and 1.0>
REASON: <Brief one sentence explanation>"""
