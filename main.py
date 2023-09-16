import ultralytics
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="AI_Dumping_Detection_Data/data.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format