import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
import os

# Cargar datos
(x_train, y_train), (_, _) = mnist.load_data()

# Filtrar solo los números del 1 al 9 (quitamos el 0)
x_train = x_train[y_train != 0]
y_train = y_train[y_train != 0]

# Normalizar las imágenes
x_train = x_train / 255.0

# Redimensionar las imágenes a 28x28x1 para CNN
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

# Aumento de datos
datagen = ImageDataGenerator(
    rotation_range=10,   # Rotación aleatoria de imágenes
    width_shift_range=0.1,  # Traslación horizontal
    height_shift_range=0.1,  # Traslación vertical
    zoom_range=0.1,      # Zoom aleatorio
    shear_range=0.2,     # Desplazamiento de perspectiva
    horizontal_flip=True, # Flip horizontal aleatorio
    fill_mode='nearest'  # Rellenar los espacios vacíos
)

# Crear el modelo CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(9, activation='softmax')
])

# Compilamos el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Ajustar el modelo con aumento de datos
model.fit(datagen.flow(x_train, y_train - 1, batch_size=32), epochs=10)

# Crear carpeta para guardar el modelo
os.makedirs("model", exist_ok=True)
model.save("model/model.h5")

# Descargar el modelo (solo en Colab)
from google.colab import files
files.download("model/model.h5")
