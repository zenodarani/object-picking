from src.robot_control import robot
import requests
import zipfile
import numpy as np
import cv2
import FANUCethernetipDriver

FANUCethernetipDriver.DEBUG = False

class vrobot(robot):
    def __init__(self, robotIP, grab_program='IMAGE_GRAB', robot_path='UD1/', zipname='IMREG1', imgname='VIMG000000.PNG'):
        # Call base class constructor 'robot'
        super().__init__(robotIP)
        self.grab_program = grab_program
        self.robot_path = robot_path
        self.zipname = zipname
        self.imgname = imgname

    # Grab image
    def grab_image(self, verbose=False):

        # Run TP program that snaps image
        FANUCethernetipDriver.writeR_Register(self.robot_IP, self.start_register, 2)
        busy = self.read_robot_start_register()
        while busy:
            busy = self.read_robot_start_register()

        zipaddress = 'http://' + self.robot_IP + '/' + self.robot_path + self.zipname + '.ZIP'

        response = requests.get(zipaddress)
        # Check download ok
        if response.status_code == 200:
            # Open file in binary writing mode
            local_path = 'temp_img/' + self.zipname
            with open(local_path, 'wb') as file:
                # Write data in a local file
                file.write(response.content)
            if verbose:
                print('Image successfully downloaded!')
            with zipfile.ZipFile(local_path, 'r') as zip_ref:
                # Extract image from zip
                with zip_ref.open(self.zipname + '/' + self.imgname) as img_file:
                    # Read binary data
                    img_data = img_file.read()
                    # Convert to numpy array
                    img_array = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    return img

        else:
            print('Image download error, html code:', response.status_code)

