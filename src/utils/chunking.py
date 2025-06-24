import aiohttp
import asyncio
import os
from typing import List

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


async def semantic_merge(text: str, word_limit: int = 512) -> List[str]:
    words = text.split()
    i = 0
    final_chunks = []

    while i < len(words):
        chunk_words = words[i : i + word_limit]
        chunk_text = " ".join(chunk_words).strip()

        prompt = f"""
You are a markdown document segmenter.

Here is a portion of a markdown document:

---
{chunk_text}
---

If this chunk clearly contains **more than one distinct topic or section**, reply ONLY with the **second part**, starting from the first sentence or heading of the new topic.

If it's only one topic, reply with NOTHING.

Keep markdown formatting intact.
"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "stream": False,
                }
                async with session.post(
                    OPENAI_CHAT_URL, headers=headers, json=payload, timeout=20
                ) as resp:
                    data = await resp.json()
                    reply = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                    )

            if reply:
                split_point = chunk_text.find(reply)
                if split_point != -1:
                    first_part = chunk_text[:split_point].strip()
                    second_part = reply.strip()
                    final_chunks.append(first_part)
                    leftover_words = second_part.split()
                    words = leftover_words + words[i + word_limit :]
                    i = 0  # reset index with new leftover
                    continue
                else:
                    final_chunks.append(chunk_text)
            else:
                final_chunks.append(chunk_text)

        except Exception as e:
            print(f"[semantic_merge] Error: {e}")
            final_chunks.append(chunk_text)

        i += word_limit

    return final_chunks
