import numpy as np
import math
import pydirectinput as dir
import time

# Configurações
screen_width, screen_height = dir.size()  # Obtem o tamanho da tela
amplitude = 100  # Amplitude do cos(x)
frequency = 4  # Frequência do cos(x)
x_range = 2 * math.pi  # Intervalo de x (0 a 2π)
resolution = 500  # Número de pontos para plotar

from PIL import Image


# Gera coordenadas a partir de uma imagem
def generate_image_coordinates(image_path):
    image = Image.open(image_path)
    image = image.convert("L")  # Convert to grayscale
    width, height = image.size
    width //= 3
    height //= 3
    coordinates = []

    for y in range(height):
        for x in range(width):
            brightness = image.getpixel((x, y))
            if brightness < 128:  # Consider dark pixels
                screen_x = int((x / width) * screen_width)
                screen_y = int((y / height) * screen_height)
                coordinates.append((screen_x, screen_y))

    image.close()
    return coordinates


# Simula o desenho com o mouse
def draw_cos_function(coordinates):
    print("Posicione o cursor na área de desenho. O programa iniciará em 5 segundos...")
    time.sleep(5)  # Tempo para o usuário posicionar o mouse

    current_position = dir.position()

    dir.mouseDown(
        coordinates[0][0] + current_position[0], coordinates[0][1] + current_position[1]
    )

    for x, y in coordinates:
        dir.moveTo(x + current_position[0], y + current_position[1])
    dir.mouseUp()  # Solta o botão após finalizar


# Executar
if __name__ == "__main__":
    coordinates = generate_image_coordinates(image_path="image.png")
    draw_cos_function(coordinates)
