# Face Recognition System

## Overview

This project provides a system for facial recognition using a pre-trained InceptionResNetV1 model. It includes functionality for adding new faces to a database, finding matches in the database for a given image, and real-time face recognition using a webcam.

## Dependencies

- Python 3.x
- TensorFlow
- NumPy
- OpenCV
- PIL (Pillow)
- Matplotlib
- JSON

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/facerecognition.git
   cd facerecognition
   Install the required packages:
   
   sh
   
   pip install numpy tensorflow opencv-python pillow matplotlib
   Download the InceptionResNetV1 weights and place them in the model directory.
   
   Usage
   FaceRecognition Class
   The FaceRecognition class provides methods for processing images, extracting facial embeddings, and managing an embeddings database.

Initialization
python
from facerecognition import FaceRecognition

model_path = 'path/to/inceptionresnetv1_weights.h5'
recognition = FaceRecognition(model_path)
Add a Face to the Database
python
recognition.add_to_db('path/to/image.jpg', 'Person Name')
Find a Face in the Database
python
Copiar código
result = recognition.find_in_db('path/to/image.jpg')
print(result)
RealTimeFaceRecognition Class
The RealTimeFaceRecognition class extends FaceRecognition to provide real-time face recognition using a webcam.

Initialization
python
from facerecognition import RealTimeFaceRecognition

model_path = 'path/to/inceptionresnetv1_weights.h5'
real_time_recognition = RealTimeFaceRecognition(model_path)
Recognize Faces from Webcam
python
real_time_recognition.recognize_from_camera()
Press q to quit the real-time recognition.

Detailed Method Descriptions
FaceRecognition Class
__init__(self, model_path: str)

Initializes the model and loads the embeddings database.
_load_image(filename: str) -> np.ndarray

Loads an image from a file and converts it to a NumPy array.
_extract_face(image: np.ndarray, required_size: Tuple[int, int] = (160, 160)) -> Optional[np.ndarray]

Detects and extracts the first face from an image, resizing it to the required size.
_get_embedding(face_pixels: np.ndarray) -> np.ndarray

Generates an embedding for a face using the InceptionResNetV1 model.
_process_image(filepath: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]

Processes an image to extract a face and generate its embedding.
_plot_face(face_array: np.ndarray) -> None

Plots a face image using Matplotlib.
_calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float

Calculates the cosine similarity between two embeddings.
add_to_db(self, image_path: str, name: str) -> None

Adds a face and its embedding to the database.
find_in_db(self, image_path: str, threshold: float = 0.5) -> Optional[str]

Finds the best match for a face in the database based on cosine similarity.
_save_db(self, filename: str = 'embeddings_db.json') -> None

Saves the embeddings database to a JSON file.
_load_db(self, filename: str = 'embeddings_db.json') -> List[dict]

Loads the embeddings database from a JSON file.
RealTimeFaceRecognition Class
__init__(self, model_path: str)

Initializes the model and loads the embeddings database, inheriting from FaceRecognition.
recognize_from_camera(self, threshold: float = 0.5) -> None

Captures frames from a webcam and performs real-time face recognition, displaying results on the screen.
Notes
Ensure that the InceptionResNetV1 weights are correctly downloaded and the path is properly set.
Adjust the threshold parameter in find_in_db and recognize_from_camera to fine-tune the matching sensitivity.
For real-time recognition, ensure that the webcam is properly connected and configured.
