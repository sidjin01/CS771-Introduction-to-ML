import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, \
 DirectoryIterator
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras import Input
from tensorflow.keras import Model
from pathlib import Path


default = Path(__file__).absolute().parent / "extracted_letters_train"
print("loading data...")
image_dataset_generator = ImageDataGenerator(rescale=1./255,
                                             rotation_range=10,
                                             validation_split=0.2)
data = DirectoryIterator(default,
                         image_dataset_generator,
                         target_size=(150, 150),
                         class_mode='categorical',
                         batch_size=64,
                         subset='training')
val_data = DirectoryIterator(default,
                             image_dataset_generator,
                             target_size=(150, 150),
                             class_mode='categorical',
                             batch_size=64,
                             subset='validation')
print("data loaded")
print("showing a sample image and its label...")
d = iter(data)
img, labil = next(d)
plt.imshow(img[3])
plt.show()

print("Preparing model...")
input = Input(shape=(150, 150, 3))
x = Conv2D(16, 3, activation='relu')(input)
x = MaxPooling2D(2)(x)
x = Conv2D(4, 3, activation='relu')(x)
x = MaxPooling2D(3)(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
output = Dense(26, activation='softmax')(x)
model = Model(inputs=input, outputs=output)
print(model.summary())
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
print("training model...")
model.fit_generator(data, validation_data=val_data, epochs=5)
print("training complete")
