from typing import List, Tuple
import cv2
import numpy as np
from numpy.typing import NDArray


class TextBox:
  """Class representing a text box identified by OCR"""

  def __init__(
    self, x1: int, y1: int, x2: int, y2: int, words: List[str], label: str = "O"
  ) -> None:
    self.x1: int = x1
    self.y1: int = y1
    self.x2: int = x2
    self.y2: int = y2
    self.words: List[str] = words
    self.label: str = label

  @property
  def text(self) -> str:
    """Return the text content of the box by joining words"""
    return " ".join(self.words)

  @property
  def coordinates(self) -> Tuple[int, int, int, int]:
    """Return the coordinates of the box as a tuple"""
    return (self.x1, self.y1, self.x2, self.y2)

  def contains_point(self, x: int, y: int) -> bool:
    """Check if a point is inside the box"""
    return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

  def draw(
    self,
    image: NDArray[np.uint8],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
  ) -> None:
    """Draw the box on an image"""
    cv2.rectangle(
      image,
      (self.x1, self.y1),
      (self.x2, self.y2),
      color,
      thickness,
    )
