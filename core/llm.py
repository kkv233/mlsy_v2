import os
import json
import time
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        self.api_key = os.environ.get("API_KEY", "")
        self.base_url = os.environ.get("BASE_URL", os.environ.get("API_BASE_URL", "https://api.openai.com/v1"))
        self.model = os.environ.get("BASE_MODEL", os.environ.get("API_MODEL", "gpt-4o"))
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.max_retries = 3
        self.retry_delay = 10
        self.request_interval = 5  # seconds between requests to avoid rate limiting
        self.last_request_time = 0

    def _wait_for_rate_limit(self):
        """Ensure minimum interval between requests to avoid rate limiting."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.request_interval:
            wait_time = self.request_interval - elapsed
            logger.info(f"  [LLM] Rate limit protection: waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        self.last_request_time = time.time()

    def chat(self, messages, temperature=0.3, max_tokens=4096, timeout=300) -> str:
        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()
                logger.info(f"  [LLM] Sending request (attempt {attempt+1}, model={self.model}, max_tokens={max_tokens})...")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                content = response.choices[0].message.content
                if not content:
                    reasoning = getattr(response.choices[0].message, 'reasoning_content', None)
                    if reasoning:
                        logger.info(f"  [LLM] content is empty, using reasoning_content ({len(reasoning)} chars)")
                        content = reasoning
                if not content:
                    logger.warning(f"  [LLM] Response content is empty! finish_reason={response.choices[0].finish_reason}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return ""
                logger.info(f"  [LLM] Response received: {len(content)} chars, finish_reason={response.choices[0].finish_reason}")
                return content
            except Exception as e:
                logger.warning(f"  [LLM] Request failed (attempt {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {e}")

    def chat_with_system(self, system_prompt: str, user_prompt: str, temperature=0.3, max_tokens=4096, timeout=300) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.chat(messages, temperature, max_tokens, timeout)

    def extract_json(self, text: str) -> dict:
        try:
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0].strip()
            else:
                json_str = text.strip()
            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            return {}
