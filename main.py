from flask import Flask,request,jsonify
from flask_cors import CORS
import os
from Network import CNN
import torch
from torchvision.transforms import transforms
from PIL import Image


model = CNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize([150,150]),
    transforms.ToTensor(),
]

)

app = Flask(__name__)
CORS(app)

folder_name = 'uploads'
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

@app.route('/predict',methods=['POST'])
def predict():

    file = request.files['file']
    if file:
        file_path = os.path.join(folder_name,file.filename)
        file.save(file_path)

        image = Image.open(file_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            _,predicted = torch.max(output,1)

        lables = ['NO PNEUMONIA','PNEUMONIA']
        prediction = lables[predicted.item()]
        return jsonify(prediction)


if __name__ == '__main__':
    app.run(debug=True)
