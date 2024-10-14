from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gc
import os
import numpy as np
import dlib
from face_lib import shape2points
from skimage import transform
from face_lib import crop_face
import cv2

# Class to use in image preprocessing

class Preprocessor:

    def __init__(self) -> None:
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.standard_model = None
        self.trans = transform.PolynomialTransform()

    #Load the dataset and store videos and labels in separate lists
    def load_dataset(dataset_path, emotion_labels):
        

        # Arrays to store videos and corresponding labels
        videos = []
        labels = []

        # Supported image formats
        supported_image_formats = ('.jpg', '.jpeg', '.png')

        # Traverse the dataset folder
        for subject_folder in os.listdir(dataset_path):
            subject_path = os.path.join(dataset_path, subject_folder)
            
            # Check if it's a directory
            if os.path.isdir(subject_path):
                # Traverse each emotion folder within each subject folder
                for emotion_folder in os.listdir(subject_path):
                    emotion_path = os.path.join(subject_path, emotion_folder)
                    
                    # Check if it's a directory and corresponds to a valid emotion
                    if os.path.isdir(emotion_path) and emotion_folder in emotion_labels:
                        # Get the numeric label for the emotion
                        label = emotion_labels[emotion_folder]
                        # Array to store the frames in one video
                        emotion_data = []
                        # Traverse all the files in the emotion folder
                        for image_file in os.listdir(emotion_path):
                            # Only process files with valid image extensions
                            if image_file.endswith(supported_image_formats):
                                image_path = os.path.join(emotion_path, image_file)
                                image = cv2.imread(image_path)
                                # Add the image to the array
                                emotion_data.append(image)

                        #Only add the folder content if it is not empty
                        if len(emotion_data) > 0:      
                            videos.append(emotion_data)
                            labels.append(label)
                                

        return videos, labels

    # Load the landmark position of the standard face model from a csv file and store them in standard_model
    def load_landmarks_from_csv(
        self,
        file_name: str
    ) -> None:
        """
        Reads landmarks from a csv file.
        Arguments
        file_name : A csv file with landmarks
        Returns
        numpy array with the landmarks
        """
        standard_model = np.zeros((68, 2))
        with open(file_name, "r") as f:
            for i, line in enumerate(f.readlines()):
                line_split = line.replace("\n", "").split(",")
                standard_model[i] = [float(value) for value in line_split]
        # Multiply with 500 (width) as the landmarks are normalized
        standard_model *= 500
        self.standard_model = standard_model

    #Function that preprocesses an image
    def preprocess(self, image):
    
        dets = self.detector(image, 1)

        #Extract the shape of the face in the first rectangle (using the first element of the rectangles variable)
        shape = self.predictor(image, dets[0])

        #Extract facial landmarks from shape by calling the shape2points() function.
        #landmarks = shape2points(shape)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        # Calculating the transorfmation between the two set of points 
        self.trans.estimate(self.standard_model, landmarks)

        # Warp the example image
        registered_img = transform.warp(image, self.trans, output_shape=(600,500))

        # Use OpenCV's warpAffine for faster warping
        # warp_matrix = cv2.getAffineTransform(
        #     np.float32(landmarks[:3]),  # Select three points from landmarks
        #     np.float32(self.standard_model[:3])
        # )
        # registered_img = cv2.warpAffine(image, warp_matrix, (500, 600))

        # Crop the face from registered image.
        cropped_registered_face = crop_face(registered_img, self.standard_model)

        return cropped_registered_face
    

    
    def preprocess_batch(self, images, batch_size=10):
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            with ProcessPoolExecutor() as executor:
                batch_results = list(executor.map(self.preprocess, batch))
            results.extend([result for result in batch_results if result is not None])
            # Free memory
            gc.collect()
        return results
        