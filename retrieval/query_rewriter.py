"""Query rewriting, expansion, and conversational context resolution."""

from typing import List, Optional, Dict, Any
from core.logger import logger
from agent.prompts import QUERY_REWRITE_PROMPT, QUERY_REWRITE_WITH_HISTORY_PROMPT


class QueryRewriter:
    """Transforms raw conversational queries into optimized retrieval queries."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def rewrite_query(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> List[str]:
        """
        Produce refined query variants (Multi-query search).
        Returns a list of search queries starting with the best rewritten query.
        """
        if not query or len(query.strip()) < 3:
            return [query]

        # Format chat history if provided
        history_str = ""
        if chat_history:
            recent_turns = chat_history[-4:]  # Last 2 exchanges
            formatted_turns = []
            for m in recent_turns:
                role = m.get("role", "user").capitalize()
                content = m.get("content", "")
                if content:
                    formatted_turns.append(f"{role}: {content}")
            history_str = "\n".join(formatted_turns)

        # If LLM client is available, invoke structured rewrite prompt
        if self.llm_client is not None:
            try:
                if history_str:
                    prompt = QUERY_REWRITE_WITH_HISTORY_PROMPT.format(
                        chat_history=history_str,
                        query=query
                    )
                else:
                    prompt = QUERY_REWRITE_PROMPT.format(query=query)

                response = self.llm_client.invoke(prompt)
                resp_text = response.content if hasattr(response, "content") else str(response)
                lines = [
                    line.strip().lstrip("123456789.-* ")
                    for line in resp_text.strip().splitlines()
                    if line.strip()
                ]
                queries = [query] + [q for q in lines if q and q.lower() != query.lower()]
                logger.info(f"Rewritten query '{query}' -> {queries}")
                return queries[:3]
            except Exception as e:
                logger.warning(f"Query rewriting failed: {e}. Using original query.")

        # Heuristic query refinement fallback
        refined_queries = [query]
        clean_q = query
        fillers = ["can you tell me", "please explain", "what is", "how do i", "i want to know", "tell me about"]
        for f in fillers:
            if clean_q.lower().startswith(f):
                clean_q = clean_q[len(f):].strip(" ?:.,")
                if clean_q:
                    refined_queries.append(clean_q)
                break

        return refined_queries
