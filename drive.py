import argparse
import base64
from datetime import datetime
import os
import sys
import csv
import shutil
import copy
from model import load_h5_model, save_h5_model
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

# initalize socketio server and flask
sio = socketio.Server()
app = Flask(__name__)

class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.
        return

    def set_desired(self, desired):
        self.set_point = desired
        return

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral

# globals used by telemetry
driver_model = None
throttle_controller = SimplePIController(0.1, 0.002)
set_speed = 30
throttle_controller.set_desired(set_speed)

@sio.on('telemetry')
def telemetry(sid, data):
    global driver_model, train_image_array, train_steering_angle_array, next_train_index
    if data:
        # car data
        steering_angle = data["steering_angle"]
        throttle = data["throttle"]
        speed = data["speed"]
        driver = data.get("driver") # driver can be "human" or "AI"
        imgString = data["image"]
        
        # process car image
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        
        # If driver is human then the AI is being manually overriden
        # Save data for training
        if driver == "human":
            sio.emit('manual', data={}, skip_sid=True)
            save_image(image, '/mnt/c/easy/live_training_data/IMG', data)
        elif driver == "AI":
            # update desired steering angle (using model + decay controller) and throttle (using PI controller)
            steering_angle = driver_model.predict(image_array[None, :, :, :], batch_size=1)
            steering_angle = float(steering_angle)
            throttle = throttle_controller.update(float(speed))
            send_control(steering_angle, throttle)
        else:
            print("ERROR: unkown driver: {0}".format(driver))
        
        print(driver, steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            save_image(image, args.image_folder)
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)
    return

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)
    return

def save_image(image, folder, data=None):
    if not os.path.exists(folder):
        os.makedirs(folder)
    timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
    image_filename = os.path.join(folder, timestamp + '.jpg')
    print(image_filename)
    image.save(image_filename)
    if data is not None:
        with open("/mnt/c/easy/live_training_data/driving_log.csv", "a") as f:
            driving_log = csv.writer(f)
            row = [image_filename, '', '', data['steering_angle'], data['throttle'], data['driver'], data['speed']]
            driving_log.writerow(row)
    return

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)
    return

def parse_args():
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    driver_model = load_h5_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if os.path.exists(args.image_folder):
            shutil.rmtree(args.image_folder)
        os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    
    print("drive.py ended")