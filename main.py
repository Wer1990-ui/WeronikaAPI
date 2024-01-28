from flask import Flask
from flask_restful import Resource, Api
import cv2

app = Flask(__name__)
api = Api(app)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


class WeronikaApi(Resource):
    def get(self):
        ing = cv2.imread('pap_20230719_1DE.jpg')
        boxes, weights = hog.detectMultiScale(ing, winStride=(8, 8))

        return {'count': len(boxes)}

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


api.add_resource(WeronikaApi, '/')
api.add_resource(HelloWorld, '/test')

if __name__ == '__main__':
    app.run(debug=True)



