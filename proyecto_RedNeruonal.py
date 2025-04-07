import numpy as np
from PIL import Image
import os
import math
from collections import deque

class RedNeuronal:
    def __init__(self, capa_entrada, capa_oculta=64, capa_salida=1):
        self.capa_entrada = capa_entrada
        self.capa_oculta = capa_oculta
        self.capa_salida = capa_salida
        valor_peso_entrada = 0.2
        valor_peso_salida = 0.5
        
        self.pesos_entrada_oculta = [[valor_peso_entrada for _ in range(capa_oculta)]
                                     for _ in range(capa_entrada)]
        self.pesos_oculta_salida = [[valor_peso_salida for _ in range(capa_salida)]
                                    for _ in range(capa_oculta)]
        self.sesgo_oculta = [0.1] * capa_oculta
        self.sesgo_salida = [0.1] * capa_salida

    def relu(self, x):
        return max(0, x)

    def derivada_relu(self, x):
        return 1 if x > 0 else 0

    def probabilidad(self, x):
        exp_x = [math.exp(i) for i in x]
        suma = sum(exp_x)
        return [i / suma for i in exp_x]

    def backpropagation(self, datos_entrada):
        entrada_oculta = [
            sum(datos_entrada[i] * self.pesos_entrada_oculta[i][j]
                for i in range(self.capa_entrada)) + self.sesgo_oculta[j]
            for j in range(self.capa_oculta)
        ]
        salida_oculta = [self.relu(x) for x in entrada_oculta]

        entrada_salida = [
            sum(salida_oculta[j] * self.pesos_oculta_salida[j][k]
                for j in range(self.capa_oculta)) + self.sesgo_salida[k]
            for k in range(self.capa_salida)
        ]
        salida = self.probabilidad(entrada_salida)

        return salida_oculta, salida

    def entrenar(self, datos_entrada, objetivo, epochs=50, batch_size=128):
        for epoch in range(epochs):
            error_total = 0
            for i in range(0, len(datos_entrada), batch_size):
                x = datos_entrada[i:i + batch_size]
                y = objetivo[i:i + batch_size]
                
                cola_datos = deque()
                min_len = min(len(x), len(y))
                for j in range(min_len):
                    cola_datos.append((x[j], y[j]))
                while cola_datos:
                    x, y = cola_datos.popleft()  
                    salida = self.backpropagation(x)
                    error = [(s - t) for s, t in self._iterar_pares(salida, y)]
                    error_total += sum([e ** 2 for e in error])

            if (epoch + 1) % 10 == 0:
                print(f"Épochs {epoch + 1}, Error: {error_total:.4f}")
    
    def _iterar_pares(self, lista1, lista2):
        resultado = []
        min_len = min(len(lista1), len(lista2))
        for i in range(min_len):
            resultado.append((lista1[i], lista2[i]))
        return resultado

    def predecir(self, datos_entrada):
        _, salida = self.backpropagation(datos_entrada)
        return salida


class ComparadoImagenes:
    def __init__(self, tamaño_objetivo=(100, 100)):
        self.tamaño_objetivo = tamaño_objetivo
        self.capa_entrada = tamaño_objetivo[0] * tamaño_objetivo[1] * 3

    def procesar_imagen(self, ruta_imagen):
        imagen = Image.open(ruta_imagen)
        imagen = imagen.resize(self.tamaño_objetivo)
        pixeles = np.array(imagen) / 255.0
        pixeles = pixeles.flatten()
        return pixeles

    def comparar_imagenes(self, ruta_imagen1, ruta_imagen2):
        pixeles1 = self.procesar_imagen(ruta_imagen1)
        pixeles2 = self.procesar_imagen(ruta_imagen2)
        diferencia = np.sqrt(np.sum((pixeles1 - pixeles2) ** 2))
        diferencia_maxima = np.sqrt(self.capa_entrada)
        similitud = 1 - (diferencia / diferencia_maxima)
        return similitud


def main():
    carpeta_base = r"C:\Users\Ricardo\Documents\Clases\IA\Proyecto"
    ruta_imagen1 = os.path.join(carpeta_base, "me3.jpg")
    ruta_imagen2 = os.path.join(carpeta_base, "yo.jpg")

    if not os.path.exists(ruta_imagen1):
        print(f"Error: No se encontró la imagen en {ruta_imagen1}")
        return

    if not os.path.exists(ruta_imagen2):
        print(f"Error: No se encontró la imagen en {ruta_imagen2}")
        return

    print(f"Analizando imágenes:")
    comparador = ComparadoImagenes(tamaño_objetivo=(100, 100))
    similitud = comparador.comparar_imagenes(ruta_imagen1, ruta_imagen2)
    print(f"\nResultado del análisis:")
    print(f"Nivel de similitud entre las imágenes: {similitud:.4f}")

    if similitud > 0.9:
        print("Las imágenes son muy similares.")
    elif similitud > 0.65:
        print("Las imágenes tienen similitud.")
    else:
        print("Las imágenes son diferentes.")

if __name__ == "__main__":
    main()