import os
from tqdm import tqdm_notebook
from keras.preprocessing import image
from pickle import dump, load

class FeatureExtractor():    
    def extract_feature(self, filename, model, app, image_size):
        # load an image from file
        img = image.load_img(filename, target_size=image_size)
        img = image.img_to_array(img)

        # reshape data for the model and preprocess for the ResNet model
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = app.preprocess_input(img)

        #get image features
        return model.predict(img, verbose=0)

    def extract_features(self, base_path, model, app, image_size, output_file):
        features = dict()

        files_list = [os.path.join(dirpath, filename) 
                      for dirpath, dirs, filenames in os.walk(base_path) 
                      for filename in filenames if not filename.startswith('.')
                     ]

        #temp variables for batch dumping features to file
        #and keeping track of already processed files 
        i = 0
        processed_files = []
        for filename in tqdm_notebook(files_list):
            if not filename.startswith('.'):

                feature = self.extract_feature(filename, model, app, image_size)

                #store feature
                image_id = filename.split('/')[-1].split('.')[0]
                features[image_id] = feature

                #update progress counter and once per 500 rounds make features dump
                i += 1
                if i%500 == 0:
                    dump(features, open('temp_features.pkl', 'wb'))

        dump(features, open(output_file, 'wb'))

        return features, processed_files
    
    def load_features(self, filename):
        with open(filename, 'rb') as features_file:
            features = load(features_file)

        return features



