import math
from typing import Self


class Vector:

    def __init__(self, x: int, y: int, z: int):
        self._x, self._y, self._z = x, y, z

    @property
    def x(self) -> int:
        return self._x

    @property
    def y(self) -> int:
        return self._y

    @property
    def z(self) -> int:
        return self._z

    @property
    def value(self) -> tuple[int, int, int]:
        return (self._x, self._y, self._z)

    @property
    def length2(self) -> int:
        return self._x * self._x + self._y * self._y + self._z * self._z

    @property
    def length(self) -> float:
        return math.sqrt(self.length2)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return (self._x, self._y, self._z) == (other._x, other._y, other._z)

    def __hash__(self) -> int:
        return hash((self._x, self._y, self._z))

    def __add__(self, other: Self) -> Self:
        return Vector(self._x + other._x, self._y + other._y, self._z + other._z)

    def __sub__(self, other: Self) -> Self:
        return Vector(self._x - other._x, self._y - other._y, self._z - other._z)

    def __mul__(self, factor: int) -> Self:
        return Vector(self._x * factor, self._y * factor, self._z * factor)

    def __rmul__(self, factor: int) -> Self:
        return Vector(self._x * factor, self._y * factor, self._z * factor)

    def __neg__(self) -> Self:
        return Vector(-self._x, -self._y, -self._z)

    def dot(self, other: Self) -> int:
        return self._x * other._x + self._y * other._y + self._z * other._z

    def cross(self, other: Self) -> Self:
        return Vector(
            self._y * other._z - self._z * other._y,
            self._z * other._x - self._x * other._z,
            self._x * other._y - self._y * other._x,
        )

    def distance2(self, other: Self) -> int:
        return (self - other).length2

    def distance(self, other: Self) -> float:
        return math.sqrt(self.distance2(other))

    def midpoint(self, other: Self) -> Self:
        return Vector(
            (self._x + other._x) // 2,
            (self._y + other._y) // 2,
            (self._z + other._z) // 2,
        )

    @staticmethod
    def min(*points: Self) -> Self:
        return Vector(
            min(point._x for point in points),
            min(point._y for point in points),
            min(point._z for point in points),
        )

    @staticmethod
    def max(*points: Self) -> Self:
        return Vector(
            max(point._x for point in points),
            max(point._y for point in points),
            max(point._z for point in points),
        )
