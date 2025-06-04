import numpy as np
import math
import pydirectinput as dir
import time

from PIL import Image

# Gera coordenadas a partir de uma imagem
def generate_image_coordinates(image_path):
    image = Image.open(image_path)
    image = image.convert("L")  # Convert to grayscale
    width, height = image.size
    width 
    height 
    coordinates = []

    for y in range(height):
        for x in range(width):
            brightness = image.getpixel((x, y))
            if brightness < 128:  # Consider dark pixels
                screen_x = int(x) 
                screen_y = int(y)
                coordinates.append((screen_x, screen_y))

    image.close()
    return coordinates


# Simula o desenho com o mouse
def draw_cos_function(coordinates):
    print("Posicione o cursor na área de desenho. O programa iniciará em 5 segundos...")
    time.sleep(5)  # Tempo para o usuário posicionar o mouse

    current_position = dir.position()

    for x, y in coordinates:
        dir.moveTo(x + current_position[0], y + current_position[1])
        dir.mouseDown()
        dir.mouseUp()  # Solta o botão após finalizar
        


# Executar
if __name__ == "__main__":
    coordinates = generate_image_coordinates(image_path="image.png")
    draw_cos_function(coordinates)
