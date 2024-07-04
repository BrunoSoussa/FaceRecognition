from model.prediction import RealTimeFaceRecognition

if __name__ == "__main__":
    '''
    Adicionar ao banco de dados
    recognizer.add_to_db("exemplo.jpeg", "Nome Exemplo")
    
    '''
    recognizer = RealTimeFaceRecognition(model_path=r"model/keras/facenet_keras.h5")
    recognizer.recognize_from_camera()