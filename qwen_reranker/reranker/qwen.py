import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

from base.reranker import RerankerBase
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from models.document import Chunk, ChunkWithScore
from tqdm import tqdm


class QwenReranker(RerankerBase):
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        instruction: str = "Determine if the document answers the query. Output only 'yes' or 'no'.",
    ):
        self._log = logging.getLogger(self.__class__.__name__)

        self._base_url = base_url
        self._model = model
        self._api_key = api_key
        self._instruction = instruction

        self._reranker_llm = self._init_llm()

    def _init_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            name="core_reranker_llm",
            base_url=self._base_url,
            model=self._model,
            api_key=self._api_key,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
            logprobs=True,
            top_logprobs=2,
            max_completion_tokens=1,
        )

    def invoke_reranker_bulk(self, query: str, docs: list[Chunk], max_workers: int = 1) -> list[ChunkWithScore]:
        reranker_outputs = [None] * len(docs)  # Pre-allocate to preserve order

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self.invoke_reranker, query, document): idx for idx, document in enumerate(docs)
            }

            # Collect results as they complete (preserves original order)
            for future in tqdm(
                as_completed(future_to_idx),
                total=len(future_to_idx),
                desc="Reranking documents",
                disable=len(docs) < 100,
            ):
                idx = future_to_idx[future]
                try:
                    chunk_w_score = future.result()
                    reranker_outputs[idx] = chunk_w_score
                except Exception as e:
                    self._log.error(f"Error processing document {idx}: {e}")
                    reranker_outputs[idx] = ChunkWithScore(**docs[idx], reranker_score=0.0)

        return reranker_outputs

    def invoke_reranker(self, query: str, doc: Chunk) -> ChunkWithScore:
        """Invoke the reranker LLM and return the probability that the document is relevant."""
        self._log.debug(f"Invoking reranker LLM for query: {len(query)} chars, doc: {len(doc)} chars")
        prompt_messages = self._format_messages(query, doc["content"])
        response = self._reranker_llm.invoke(prompt_messages)

        yes_prob, no_prob = self._extract_yes_no_logprobs(response)
        fallback_answer = self._extract_text_answer(response)

        yes_probability = self._compute_yes_probability(yes_prob, no_prob, fallback_answer)
        return ChunkWithScore(**doc, reranker_score=yes_probability)

    def _format_messages(self, query: str, doc: str) -> list[dict]:
        return [
            {
                "role": "system",
                "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Only reply 'yes' or 'no'.",  # noqa: E501
            },
            {"role": "user", "content": f"<Instruct>: {self._instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"},
        ]

    def _extract_yes_no_logprobs(self, response: BaseMessage) -> tuple[float | None, float | None]:
        """Extract log probabilities for 'Yes' and 'No' tokens from response metadata."""
        yes_prob = None
        no_prob = None

        logprobs: dict = (response.response_metadata or {}).get("logprobs", {})
        content_logprobs: list[dict] = logprobs.get("content", [])

        if content_logprobs:
            top_logprobs: list[dict] = content_logprobs[0].get("top_logprobs", [])
            for token_info in top_logprobs:
                token = token_info.get("token", "").lower()
                logprob = token_info.get("logprob", 0.0)
                if token == "yes":
                    yes_prob = logprob
                elif token == "no":
                    no_prob = logprob

        return yes_prob, no_prob

    def _extract_text_answer(self, response: BaseMessage) -> bool:
        """Extract fallback yes/no answer from the text content of the response."""
        content = ""

        if isinstance(response.content, str):
            content = response.content.strip().lower()
        elif isinstance(response.content, list) and response.content:
            # Handle list of message chunks (e.g. in some LLM providers)
            first_chunk = response.content[0]
            content = (first_chunk.text if hasattr(first_chunk, "text") else str(first_chunk)).strip().lower()
        else:
            self._log.warning(
                f"Unexpected response content type: {type(response.content)}, content: {response.content}"
            )

        return "yes" in content

    def _compute_yes_probability(
        self,
        yes_prob: float | None,
        no_prob: float | None,
        fallback_is_yes: bool,
    ) -> float:
        """Compute the final probability of 'yes' using logprobs if available, otherwise fallback."""
        if yes_prob is not None and no_prob is not None:
            # Softmax over yes/no
            exp_yes = math.exp(yes_prob)
            exp_no = math.exp(no_prob)
            return exp_yes / (exp_yes + exp_no)

        if yes_prob is not None:
            # Only yes logprob available → assume no is very unlikely
            exp_yes = math.exp(yes_prob)
            exp_no = math.exp(no_prob or -100)  # -100 is a safe "almost zero" logprob
            return exp_yes / (exp_yes + exp_no)

        # No logprobs → fall back to text parsing
        return 1.0 if fallback_is_yes else 0.0
