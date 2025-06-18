indexed_url_memory = set()


def is_url_already_indexed(url: str) -> bool:
    return url in indexed_url_memory


def register_indexed_url(url: str):
    indexed_url_memory.add(url)
