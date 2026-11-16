"""Data access: corpus posts, questions, and the reference results.

All reads are cached in-process (the bundled data never changes at runtime),
so repeated API calls do not re-read the disk.
"""

import json
from functools import lru_cache
from pathlib import Path

from app.core.config import settings
from app.models.corpus import Question, Rewording
from app.services.text_service import clean_markdown


class CorpusRepository:
    """Loads the bundled corpus, eval questions, and reference results."""

    def __init__(
        self,
        corpus_dir: Path | None = None,
        questions_file: Path | None = None,
        results_file: Path | None = None,
    ) -> None:
        self.corpus_dir = corpus_dir or settings.corpus_dir
        self.questions_file = questions_file or settings.questions_file
        self.results_file = results_file or settings.results_file

    @lru_cache(maxsize=1)
    def load_posts(self) -> dict[str, dict]:
        """Load every ``*.md`` post: ``{slug: {title, text, words}}``.

        Mirrors the original script: the title is the file stem, the text is
        the markdown-cleaned body, and ``words`` counts whitespace-separated
        words of the cleaned text.
        """
        posts: dict[str, dict] = {}
        for path in sorted(self.corpus_dir.glob("*.md")):
            text = clean_markdown(path.read_text(encoding="utf-8"))
            posts[path.stem] = {
                "title": path.stem,
                "text": text,
                "words": len(text.split()),
            }
        return posts

    @lru_cache(maxsize=1)
    def load_questions(self) -> list[Question]:
        """Load the eval questions from ``questions.json``."""
        data = json.loads(self.questions_file.read_text(encoding="utf-8"))
        return [Question(**item) for item in data["questions"]]

    @lru_cache(maxsize=1)
    def load_rewordings(self) -> list[Rewording]:
        """Load the rewording variants from ``questions.json``."""
        data = json.loads(self.questions_file.read_text(encoding="utf-8"))
        return [Rewording(**item) for item in data["rewordings"]]

    @lru_cache(maxsize=1)
    def load_reference_results(self) -> dict:
        """Load the reference results.json from the original script."""
        return json.loads(self.results_file.read_text(encoding="utf-8"))
