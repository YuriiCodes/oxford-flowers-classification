import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16

from tensorflow.keras.models import Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import tensorflow_datasets as tfds

# Завантаження датасету
dataset, info = tfds.load("oxford_flowers102", with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

def preprocess_image(image, label):
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0  # нормалізація пікселів до діапазону 0–1
    return image, label


train_data = train_dataset.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)
test_data = test_dataset.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)

# plt.figure(figsize=(10, 10))
for images, labels in train_data.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy())
        plt.title(int(labels[i]))
        plt.axis("off")
plt.show()

# Відсоток даних для валідації (наприклад, 15%)
val_split = 0.15

# Розрахунок кількості даних для валідації
total_train = tf.data.experimental.cardinality(train_dataset).numpy()
val_size = int(total_train * val_split)

# Виділяємо дані для валідації
val_dataset = train_dataset.take(val_size)
train_dataset = train_dataset.skip(val_size)


# Застосування функції обробки зображень
train_data = train_dataset.map(preprocess_image).batch(32).shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_data = val_dataset.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)
test_data = test_dataset.map(preprocess_image).batch(32).prefetch(tf.data.AUTOTUNE)


# Перевірка оброблених зображень з тренувального набору
plt.figure(figsize=(10, 10))
for images, labels in train_data.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy())
        plt.title(int(labels[i]))
        plt.axis("off")
plt.show()




# Завантажуємо модель VGG16 з переднавченими вагами на ImageNet
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Заморожуємо шари базової моделі, щоб вони не навчалися заново
for layer in base_model.layers:
    layer.trainable = False

# Додаємо власні шари поверх VGG16
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)  # Додатковий щільний шар для кращого узагальнення
x = Dense(102, activation='softmax')(x)  # Кінцевий шар для 102 класів квітів (Oxford Flowers має 102 види)

# Створюємо модель
model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, validation_data=val_data, epochs=10)

test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test accuracy of the pre-trained model: {test_accuracy:.2f}")



# Модель з нуля
model_from_scratch = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(102, activation='softmax')  # 102 класи для Oxford Flowers
])

model_from_scratch.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Навчання моделі
history_scratch = model_from_scratch.fit(train_data, validation_data=val_data, epochs=10)

# Оцінка моделі
test_loss_scratch, test_accuracy_scratch = model_from_scratch.evaluate(test_data)
print(f"Test accuracy of the model trained from scratch: {test_accuracy_scratch:.2f}")

# Графік точності
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Pre-trained Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Pre-trained Val Accuracy')
plt.plot(history_scratch.history['accuracy'], label='From Scratch Train Accuracy')
plt.plot(history_scratch.history['val_accuracy'], label='From Scratch Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Comparison')
plt.show()

# Графік втрат
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Pre-trained Train Loss')
plt.plot(history.history['val_loss'], label='Pre-trained Val Loss')
plt.plot(history_scratch.history['loss'], label='From Scratch Train Loss')
plt.plot(history_scratch.history['val_loss'], label='From Scratch Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Comparison')
plt.show()

