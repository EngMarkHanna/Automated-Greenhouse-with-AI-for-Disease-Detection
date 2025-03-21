
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
from io import BytesIO
from PIL import Image
import socket
import numpy as np
import tensorflow as tf
import os
import requests

CLASS_NAMES = [  ["Anthracnose Disease", "Bacterial Wilt Disease", "Downy Mildew Disease", "Fresh Leaf Healthy", "Gummy Stem Blight Disease"],
               ["Leaf scorch Disease", "Healthy"],
               ["Bacterial spot Disease", "Healthy"],
               ["Healthy", "Inspect Pest Disease", "Mosaic Virus Disease", "Small Leaf Disease", "White Mold Disease", "Wilt Disease"],
               ["Bacterial spot Disease", "Early blight Disease", "Late blight Disease", "Leaf mold Disease", "Septoria leaf spot Disease", "Yellow leaf curl Virus", "Healthy"],
               ["Early blight Disease", "Healthy", "Late blight Disease"]]
MODELS =["C:/Users/Central Intelligence/Desktop/Cucumber.h5",
         "C:/Users/Central Intelligence/Desktop/Strawberry.h5",
         "C:/Users/Central Intelligence/Desktop/Bell pepper.h5",
         "C:/Users/Central Intelligence/Desktop/Eggplant.h5",
         "C:/Users/Central Intelligence/Desktop/Tomato.h5"]


mainServerUrl = 'http://192.168.154.68:3002' #Main server URL  

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    finally:
        s.close()
    return local_ip



class Server(BaseHTTPRequestHandler):
    def do_POST(self):
        
        if '/upload' in self.path:  # Check if '/upload' is in the path
            
            modelNumber = -1
            
            if self.path == '/upload/Cucumber':  # Corrected the path
                # Handle the Cucumber case
                modelNumber = 0
            elif self.path == '/upload/Strawberry':  # Corrected the path
                # Handle the Strawberry case
                modelNumber = 1
            elif self.path == '/upload/Bell pepper':  # Corrected the path
                # Handle the Bell pepper case
                modelNumber = 2
            elif self.path == '/upload/Eggplant':  # Corrected the path
                # Handle the Eggplant case
                modelNumber = 3
            elif self.path == '/upload/Tomato':  # Corrected the path
                # Handle the Tomato case
                modelNumber = 4
            else:
                # Handle any other case
                self.send_response(404)  # Return HTTP 404 for unknown paths
                self.end_headers()
                
            if modelNumber != -1:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)  # Read the incoming data
                image = Image.open(BytesIO(post_data))  # Open the image from the received data
                image.save("C:/Users/Central Intelligence/Desktop/save images/received_image.jpg")  # Save the image to a file
                print("Image received and saved as 'received_image.jpg'")
                self.send_response(200)  # HTTP 200 indicates success
                self.end_headers()
                
                
                self.process_image_and_predict(modelNumber,image)
            
    def process_image_and_predict(self, modelNumber, image):
        # Normalize and resize the image
        image_array = np.array(image)  # Convert to NumPy array
        print(image_array.shape)
        image_array = image_array / 255.0  # Normalize pixel values to [0, 1]

        # Resize the image to 299x299
        image_array = tf.image.resize(image_array, [299, 299])
        
        # Ensure the image has a batch dimension for model prediction
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Load the model
        model = tf.keras.models.load_model(MODELS[modelNumber])

        # Make predictions
        predictions = model.predict(image_array)

         # Get the index of the maximum value in the softmax output
        max_index = np.argmax(predictions[0])  # Predictions could be a batch, hence [0]

        # Get the corresponding class name and confidence
        predicted_class = CLASS_NAMES[modelNumber][max_index]  # Retrieve the class name
        confidence = predictions[0][max_index]  # Retrieve the confidence value

        print(predicted_class, confidence * 100)
        
        # Data to send in the POST request
        
        data = {
            'predicted_class': predicted_class,  
            'confidence': confidence * 100 
        }
        
        # Send the POST request with the data as JSON
        response = requests.post(mainServerUrl, json=data)

        # Check the response
        if response.status_code == 200:
            print("Successfully sent data")
            print("Server response:", response.json())  # Assuming the server responds with JSON
        else:
            print("Failed to send data")
            print("Error:", response.status_code, response.text)
        
        return predicted_class, confidence


def run(server_class=HTTPServer, handler_class=Server, port=8000):
    ip_address = get_local_ip()
    print(ip_address)
    
    server_address = (ip_address, port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd server on port {port}')
    httpd.serve_forever()
    
print("Current working directory:", os.getcwd())  
run()


"""         if self.path == '/upload':  # Ensure the endpoint is '/upload'
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)  # Read the incoming data
            image = Image.open(BytesIO(post_data))  # Open the image from the received data
            image.save("C:/Users/Central Intelligence/Desktop/save images/received_image.jpg")  # Save the image to a file
            print("Image received and saved as 'received_image.jpg'")
            self.send_response(200)  # HTTP 200 indicates success
            self.end_headers()
            
            
            self.process_image_and_predict(1,image)
            
        else:
            self.send_response(404)  # If the endpoint is incorrect, return 404
            self.end_headers() """