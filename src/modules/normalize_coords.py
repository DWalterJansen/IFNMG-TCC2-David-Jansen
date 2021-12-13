from typing import Tuple, Union
import math


def normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
            '''Converte coordenadas normalizadas para coordenadas de pixels'''

            # Verifica se o valor normalizado está entre 0 e 1.
            def is_valid_normalized_value(value: float) -> bool:
                return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

            if not (is_valid_normalized_value(normalized_x) and
                    is_valid_normalized_value(normalized_y)):
                # TODO: Draw coordinates even if it's outside of the image bounds.
                return None
            x_px = min(math.floor(normalized_x * image_width), image_width - 1)
            y_px = min(math.floor(normalized_y * image_height), image_height - 1)
            return x_px, y_px