import numpy as np
import json
import torch


class StateVector:

    def __init__(self, x_center, y_center, DT):
        self.at_rest = False
        self.prev_ball = None
        self.prev_zero = None  # Predictive model network
        self.x_center = x_center # center of wheel hardcoded?
        self.y_center = y_center # center of wheel hardcoded?
        self.DT = DT
        self.b = 0 # ball detection count
        self.z = 0 # zero pocket detection count
        self.i = 0 # total detection cout

    def calculate_realtime(self, detections):
        ball_flag = False
        zero_flag = False
        frame_state_vector = []

        if len(detections) > 0:
            for val in detections:
                if val["cls"] == 0:
                    ball_object = val # ball is key value pairs object {'cls': 0, 'cnf': '0.92', 'x': 1073.5, 'y': 660.0}
                    ball_flag = True
                elif val["cls"] == 1:
                    zero_object = val # zero is key value pairs object {'cls': 1, 'cnf': '0.92', 'x': 1073.5, 'y': 660.0}
                    zero_flag = True
                else:
                    assert False, "Malformed class type in json array, class should be 0 or 1 only"
            self.i = self.i + 1  # Increment total detection count

            # CALCULATE BALL VELOCITES AND ACCELERATIONS
            # If there was a previous ball detection AND ball_flag = True
            if self.prev_ball is not None and ball_flag:
                # Convert cartesian coords to polar coords of ball
                ball_object = StateVector.convert_to_polar(ball_object, self.x_center, self.y_center)

                # Calculate angular velocity of ball DEGREES/SEC
                ball_object['w'] = (StateVector.calculate_radial_distance(ball_object['theta'], self.prev_ball['theta']) / (self.DT*(self.i - self.b)))
                ball_object['s'] = ((ball_object['x'] ** 2 + ball_object['y'] ** 2) ** (1 / 2)) * ball_object['w']
                ball_object.pop('x')
                ball_object.pop('y')

                # If previous ball detection has angular velocity
                if self.prev_ball['w'] is not None:
                    ball_object['a'] = ((ball_object['w'] - self.prev_ball['w']) / (self.DT*(self.i - self.b)))

                # If previous ball acceleration and velocity is not null
                if ball_object['a'] and ball_object['w'] is not None:
                    # APPEND TO VECTOR
                    frame_state_vector.append(ball_object)

                self.prev_ball = ball_object # save values of ball_object to instance object prev_ball
                self.b = self.b + 1  # increment ball detection count

            # If there was NOT a previous ball and ball_flag = True
            elif self.prev_ball is None and ball_flag:
                self.prev_ball = StateVector.convert_to_polar(ball_object, self.x_center, self.y_center)
                self.b = self.b + 1  # increment ball detection count

            #  CALCULATE ZERO SPEEDS AND ACCELERATIONS
            if self.prev_zero is not None and zero_flag:
                zero_object = StateVector.convert_to_polar(zero_object, self.x_center, self.y_center)
                zero_object['w'] = ((StateVector.calculate_radial_distance(zero_object['theta'], self.prev_zero['theta'])) / (self.DT*(self.i - self.z)))
                zero_object['s'] = ((zero_object['x'] ** 2 + zero_object['y'] ** 2) ** (1 / 2)) * zero_object['w']
                zero_object.pop('x')
                zero_object.pop('y')

                if self.prev_zero['w'] is not None:
                    zero_object['a'] = ((zero_object['w'] - self.prev_zero['w']) / ((self.i - self.z)*self.DT))

                if zero_object['a'] and zero_object['w'] is not None:
                    # APPEND TO VECTOR
                    frame_state_vector.append(zero_object)
                self.prev_zero = zero_object
                self.z = self.z + 1

            elif self.prev_zero is None and zero_flag:
                self.prev_zero = StateVector.convert_to_polar(zero_object, self.x_center, self.y_center)
                self.z = self.z + 1
        else:
            self.i = self.i + 1

        print("frame state vector length: {}".format(len(frame_state_vector)))

        if len(frame_state_vector) == 2:
            if frame_state_vector[0]['s'] - frame_state_vector[1]['s'] < 0:
                self.at_rest = True
            #return torch.tensor([frame_state_vector[0]['radius'], frame_state_vector[0]['theta'], frame_state_vector[0]['w'], frame_state_vector[0]['acceleration'], frame_state_vector[1]['radius'], frame_state_vector[1]['theta'], frame_state_vector[1]['w'], frame_state_vector[1]['acceleration']])
            return frame_state_vector
        else:
            return None


    @staticmethod
    def convert_to_polar(detection_object, x_center, y_center):

        theta = 0
        #  Calculate midpoint
        #try:
            #x = detection_object.pop('x') - x_center # subtract "x": key pair in dict object from x_center
            #y = detection_object.pop('y') - y_center # subtract "y": key pair in dict object from y_center

        x = detection_object['x'] - x_center # subtract "x": key pair in dict object from x_center
        y = detection_object['y'] - y_center # subtract "y": key pair in dict object from y_center

        radius = (x ** 2 + y ** 2) ** (1 / 2) # calculate radius or distance between two points

        if x > 0:
            if y >= 0:
                theta = np.degrees(np.arctan(y / x))
            else:
                theta = np.degrees(np.arctan(y / x)) + 360
        elif x < 0:
                theta = np.degrees(np.arctan(y / x)) + 180
        elif x == 0:
            if y > 0:
                theta = 90
            elif y < 0:
                theta = 270

        detection_object['r'] = radius
        detection_object['theta'] = theta
        detection_object['w'] = None
        detection_object['a'] = None
        detection_object['at_rest'] = False

        return detection_object # key/value pair object {'cls': 0, 'cnf': '0.95', 'radius': 0.0, 'theta': 0, 'w': None, 'a': None}
        #except KeyError:
        #   return detection_object

    @staticmethod
    def calculate_radial_distance(curr_theta, prev_theta):  # Calculates radial distance between two points
        angle = curr_theta - prev_theta
        angle = (angle + 180) % 360 - 180
        return angle


    @staticmethod
    def pv2sv(filein, fileout, x_center, y_center, result, DT):
        spin_file = open(fileout, 'a+')
        last_ball, last_zero = None, None
        b, z, i = 0, 0, 0
        pos_file = open(filein, 'r')
        det = pos_file.readline()
        while det is not None:
            ball_flag = False
            zero_flag = False
            frame_state_vector = []
            if len(det) > 0:
                json_detection = json.loads(det)
            else:
                print("File finished")
                break
            if len(json_detection) > 0:
                for inst in json_detection:
                    if inst["cls"] == 0:
                        ball = inst
                        ball_flag = True
                    elif inst["cls"] == 1:
                        zero = inst
                        zero_flag = True
                    else:
                        assert False, "Malformed class type in json array, class should be 0 or 1 only"
                i = i + 1  # increment total count

                #  Calculate ball speeds and accelerations
                if last_ball is not None and ball_flag:
                    ball = StateVector.convertToPolar(ball, x_center, y_center)
                    ball['w'] = (StateVector.calcRadialDistance(ball['theta'], last_ball['theta']) / (DT*(i - b)))
                    if last_ball['w'] is not None:
                        ball['a'] = ((ball['w'] - last_ball['w']) / (DT*(i - b)))

                    if ball['a'] and ball['w'] is not None:
                        frame_state_vector.append(ball)

                    last_ball = ball
                    b = b + 1  # increment ball detection count
                elif last_ball is None and ball_flag:
                    last_ball = StateVector.convertToPolar(ball, x_center, y_center)
                    b = b + 1  # increment ball detection count

                #  Calculate zero speeds and accelerations
                if last_zero is not None and zero_flag:
                    zero = StateVector.to_polar(zero, x_center, y_center)
                    zero['w'] = ((StateVector.rad_dist(zero['theta'], last_zero['theta'])) / (DT*(i - z)))
                    if last_zero['w'] is not None:
                        last_zero['a'] = ((zero['w'] - last_zero['w']) / ((i - z)*DT))
                    if zero['a'] and zero['w'] is not None:
                        frame_state_vector.append(zero)
                    last_zero = zero
                    z = z + 1
                elif last_zero is None and zero_flag:
                    last_zero = StateVector.to_polar(zero, x_center, y_center)
                    z = z + 1
            else:
                i = i + 1

            if len(frame_state_vector) == 2:
                spin_file.write(json.dumps(frame_state_vector) + "|"+str(result)+"\n")
            det = pos_file.readline()
