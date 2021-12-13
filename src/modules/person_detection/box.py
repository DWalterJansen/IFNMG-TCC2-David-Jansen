from typing import Tuple

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

class Box:    
    def __init__(self, p1: Point, p2: Point):
        self.p1 = p1
        self.p2 = p2


