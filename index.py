import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Carregar o conjunto de dados MNIST
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Redimensionar e normalizar as imagens
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images, test_images = train_images / 255.0, test_images / 255.0

# Definir as classes
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Construir o modelo
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Configurar o TensorBoard
logdir = 'log'
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

# Treinar o modelo
model.fit(train_images, train_labels, epochs=5, 
          validation_data=(test_images, test_labels),
          callbacks=[tensorboard_callback])

# Avaliar o modelo
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# Fazer previsões
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Gerar a matriz de confusão
confusion_matrix = tf.math.confusion_matrix(test_labels, predicted_labels)

# Visualizar a matriz de confusão
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Gerar a matriz de confusão
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Extrair VP, VN, FP, FN
VP = np.diag(conf_matrix)
FP = conf_matrix.sum(axis=0) - VP
FN = conf_matrix.sum(axis=1) - VP
VN = conf_matrix.sum() - (VP + FP + FN)

# Calcular as métricas
sensibilidade = VP / (VP + FN)
especificidade = VN / (FP + VN)
acuracia = (VP + VN) / conf_matrix.sum()
precisao = VP / (VP + FP)
f_score = 2 * (precisao * sensibilidade) / (precisao + sensibilidade)

# Exibir as métricas
for i in range(len(classes)):
    print(f'Classe {classes[i]}:')
    print(f'Sensibilidade: {sensibilidade[i]:.2f}')
    print(f'Especificidade: {especificidade[i]:.2f}')
    print(f'Acurácia: {acuracia[i]:.2f}')
    print(f'Precisão: {precisao[i]:.2f}')
    print(f'F-score: {f_score[i]:.2f}')
    print('-' * 30)