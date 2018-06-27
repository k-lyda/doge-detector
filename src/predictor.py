import numpy as np
import keras
import imageio
import matplotlib.pyplot as plt
from skimage.transform import resize

class Predictor():
    def __init__(self, name, model, base_model, app, image_size, labels_coding):
        self.name = name
        self.model = model
        self.base_model = base_model
        self.app = app
        self.image_size = image_size
        self.labels_coding = labels_coding
        
    def resize_to_max(self, img, max_dim=1000):
        h, w = img.shape[0:2]
        if h >= max_dim and w <= h:
            size = (max_dim, max_dim * w / h)
            return resize(img, size, preserve_range=True).astype('uint8')
        elif h <= w and w >= max_dim:
            size = (max_dim * h / w, max_dim)
            return resize(img, size, preserve_range=True).astype('uint8')

    def get_prediction(self, img, return_best=False):
        def get_key_by_value(d, value):
            for k, v in d.items():
                if v == value:
                    return k
        
        #preprocess image
        img = resize(img, self.image_size, preserve_range=True).astype('uint8')
        img = keras.preprocessing.image.img_to_array(img)

        # reshape data for the model and preprocess for the model
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = self.app.preprocess_input(img)
        
        #get image features
        features = self.base_model.predict(img, verbose=0)
        pred = self.model.predict(features)[0]

        if return_best:
            class_id = np.argmax(pred, axis=None, out=None)
            return [get_key_by_value(self.labels_coding, class_id), pred[class_id]]
        else: 
            #return scores above 0.1 as array of label and probability, or if there are no such probas, return top 3
            if np.any(pred > 0.1):
                probas = zip(np.where(pred > 0.1)[0], np.take(pred, np.where(pred > 0.1)[0]))
            else:
                highest_indexes = pred.argsort()[-3:][::-1]
                probas = zip(highest_indexes, np.take(pred, highest_indexes))

            labeled_probas = [(get_key_by_value(self.labels_coding, k),v) 
                           for k,v in probas
                          ]
            labeled_probas.sort(key=lambda x: x[1], reverse=True)

            return labeled_probas

    def predict_for_file(self, file_path, detector=None, verbose=0, return_best_only=False):
        predictions = []
        
        if verbose: print("\nPredictor: {}\nLoading image {}...".format(self.name, file_path))
        img = imageio.imread(file_path, pilmode='RGB')
        if verbose: print("Loaded.")
        if detector:
            if verbose: print("Detecting dogs...")
            boxes, classes, scores = detector.detect(img)
            
            dogs = []
            for i, cls in enumerate(classes):
                if detector.all_classes[cls] == 'dog':
                    dogs.append(i)

            if len(dogs) > 0:
                if verbose:
                    if len(dogs) > 1: 
                        print("{} dogs has been found. Detecting breed for each...".format(len(dogs))) 
                    else:
                        print("The dog has been found. Detecting breed...")
                for i, dog in enumerate(dogs):
                    if verbose: print("Dog {}. Detecting breed...".format(i+1))
                    x, y, w, h = boxes[dog]
                    top = max(0, np.floor(x + 0.5).astype(int))
                    left = max(0, np.floor(y + 0.5).astype(int))
                    right = min(img.shape[1], np.floor(x + w + 0.5).astype(int))
                    bottom = min(img.shape[0], np.floor(y + h + 0.5).astype(int))

                    dog_img = img[left:bottom,top:right]

                    pred = self.get_prediction(dog_img, return_best=return_best_only)
                    predictions.append(pred)
                    if verbose: print("Done.")
                        
                if verbose: 
                    boxes = [boxes[dog] for dog in dogs]
                    breeds = [predictions[i][0][0] for i in range(len(dogs))]
                    scores = [predictions[i][0][1] for i in range(len(dogs))]
                    detector.draw(img, boxes, scores, breeds)
            else:
                if verbose: print("Cannot found dogs in the picture. Are you sure that there is any?")
        else:
            if verbose: print("No detector. Detecting breed...")
            
            predictions.append(self.get_prediction(img, return_best=return_best_only))
            if verbose: 
                plt.imshow(img)
                plt.show()
                print('class: {0}, score: {1:.2f}'.format(predictions[0][0][0], predictions[0][0][1]))
                print("Done.")

            
        return predictions