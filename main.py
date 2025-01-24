import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

class BreastCancerClassifier:
    def __init__(self, base_path="D:/ML 1.1/1.1.2"):
        self.base_path = base_path
        self.image_size = (224, 224)  
        self.batch_size = 32
        
    def load_data(self):
        
        self.calc_train = pd.read_csv(os.path.join(self.base_path, 'calc_case_description_train_set.csv'))
        self.calc_test = pd.read_csv(os.path.join(self.base_path, 'calc_case_description_test_set.csv'))
        self.mass_train = pd.read_csv(os.path.join(self.base_path, 'mass_case_description_train_set.csv'))
        self.mass_test = pd.read_csv(os.path.join(self.base_path, 'mass_case_description_test_set.csv'))
        
        self.train_data = pd.concat([self.calc_train, self.mass_train])
        self.test_data = pd.concat([self.calc_test, self.mass_test])
        
        self.prepare_image_data()
        
    def prepare_image_data(self):
        def get_image_paths_and_labels(data):
            image_paths = []
            labels = []
            
            for _, row in data.iterrows():
                
                if pd.notna(row.get('image file path')):
                    dir_path = row['image file path'].split('/')[0]
                    full_dir_path = os.path.join(self.base_path, 'jpeg', dir_path)
                    
                    if os.path.exists(full_dir_path):
                        
                        jpg_files = [f for f in os.listdir(full_dir_path) if f.endswith('.jpg')]
                        for jpg_file in jpg_files:
                            image_paths.append(os.path.join(full_dir_path, jpg_file))
                            
                            label = 1 if 'MALIGNANT' in str(row.get('pathology', '')).upper() else 0
                            labels.append(label)
            
            return image_paths, labels
        
        
        self.train_paths, self.train_labels = get_image_paths_and_labels(self.train_data)
        self.test_paths, self.test_labels = get_image_paths_and_labels(self.test_data)
        
    def create_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        self.model = model
        
    def create_data_generators(self):
        
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        self.test_datagen = ImageDataGenerator(rescale=1./255)
        
    def train_model(self, epochs=10):
        
        self.create_data_generators()
        
    
        self.create_model()
        
        history = self.model.fit(
            self.train_datagen.flow_from_directory(
                self.base_path,
                target_size=self.image_size,
                batch_size=self.batch_size,
                class_mode='binary'),
            epochs=epochs,
            validation_data=self.test_datagen.flow_from_directory(
                self.base_path,
                target_size=self.image_size,
                batch_size=self.batch_size,
                class_mode='binary')
        )
        
        return history
    
    def 
        img = load_img(image_path, target_size=self.image_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.
        
        prediction = self.model.predict(img_array)
        return prediction[0][0]
    
    def save_model(self, filepath):
        self.model.save(filepath)
        
    def load_saved_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)


def main():
  
    classifier = BreastCancerClassifier()
    
    classifier.load_data()
  
    history = classifier.train_model(epochs=1)
    
    
    classifier.save_model('breast_cancer_model.h5')
    
 
    sample_image_path = "path_to_test_image.jpg"
    if os.path.exists(sample_image_path):
        prediction = classifier.predict(sample_image_path)
        print(f"Prediction for {sample_image_path}: {prediction}")
        print("1 indicates malignant, 0 indicates benign")

if __name__ == "__main__":
    main()

