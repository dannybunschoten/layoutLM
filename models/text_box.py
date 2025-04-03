import cv2

class TextBox:
    """Class representing a text box identified by OCR"""

    def __init__(self, x, y, w, h, words, label="O"):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.words = words if isinstance(words, list) else [words]
        self.label = label

    @property
    def text(self):
        """Return the text content of the box by joining words"""
        return " ".join(self.words)

    @property
    def coordinates(self):
        """Return the coordinates of the box as a tuple"""
        return (self.x, self.y, self.w, self.h)

    def contains_point(self, x, y):
        """Check if a point is inside the box"""
        return (self.x <= x <= self.x + self.w) and (self.y <= y <= self.y + self.h)

    def draw(self, image, color=(0, 255, 0), thickness=2):
        """Draw the box on an image"""
        cv2.rectangle(
            image,
            (self.x, self.y),
            (self.x + self.w, self.y + self.h),
            color,
            thickness,
        )