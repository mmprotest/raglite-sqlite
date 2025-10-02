from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from ..typing import ParsedBlock


class BaseParser(ABC):
    @abstractmethod
    def parse(self, path: str, **options: object) -> Iterable[ParsedBlock]:
        ...
