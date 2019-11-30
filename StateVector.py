import numpy as np
import json
import torch


class StateVector:

    def __init__(self, x_center, y_center, DT):
        self.prev_ball = None
        self.prev_zero = None  # Predictive model network
        self.x_center = x_center # center of wheel hardcoded?
        self.y_center = y_center # center of wheel hardcoded?
        self.DT = DT
        self.b = 0 # ball detection count
        self.z = 0 # zero pocket detection count
        self.i = 0 # total detection cout

    def calculateRealtime(self, detections):
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
                ball_object = StateVector.convertToPolar(ball_object, self.x_center, self.y_center)

                # Calculate angular velocity of ball DEGREES/SEC
                ball_object['angular velocity'] = (StateVector.calcRadialDistance(ball_object['theta'], self.prev_ball['theta']) / (self.DT*(self.i - self.b)))
                ball_object['speed'] = ((ball_object['x'] ** 2 + ball_object['y'] ** 2) ** (1 / 2)) * ball_object['angular velocity']
                ball_object.pop('x')
                ball_object.pop('y')

                # If previous ball detection has angular velocity
                if self.prev_ball['angular velocity'] is not None:
                    ball_object['acceleration'] = ((ball_object['angular velocity'] - self.prev_ball['angular velocity']) / (self.DT*(self.i - self.b)))

                # If previous ball acceleration and velocity is not null
                if ball_object['acceleration'] and ball_object['angular velocity'] is not None:
                    frame_state_vector.append(ball_object)

                self.prev_ball = ball_object # save values of ball_object to instance object prev_ball
                self.b = self.b + 1  # increment ball detection count

            # If there was NOT a previous ball and ball_flag = True
            elif self.prev_ball is None and ball_flag:
                self.prev_ball = StateVector.convertToPolar(ball_object, self.x_center, self.y_center)
                self.b = self.b + 1  # increment ball detection count

            #  CALCULATE ZERO SPEEDS AND ACCELERATIONS
            if self.prev_zero is not None and zero_flag:
                zero_object = StateVector.convertToPolar(zero_object, self.x_center, self.y_center)
                zero_object['angular velocity'] = ((StateVector.calcRadialDistance(zero_object['theta'], self.prev_zero['theta'])) / (self.DT*(self.i - self.z)))

                if self.prev_zero['angular velocity'] is not None:
                    zero_object['acceleration'] = ((zero_object['angular velocity'] - self.prev_zero['angular velocity']) / ((self.i - self.z)*self.DT))

                if zero_object['acceleration'] and zero_object['angular velocity'] is not None:
                    frame_state_vector.append(zero_object)
                self.prev_zero = zero_object
                self.z = self.z + 1

            elif self.prev_zero is None and zero_flag:
                self.prev_zero = StateVector.convertToPolar(zero_object, self.x_center, self.y_center)
                self.z = self.z + 1
        else:
            self.i = self.i + 1

        print("frame state vector length: {}".format(len(frame_state_vector)))

        if len(frame_state_vector) == 2:
            #return torch.tensor([frame_state_vector[0]['radius'], frame_state_vector[0]['theta'], frame_state_vector[0]['w'], frame_state_vector[0]['acceleration'], frame_state_vector[1]['radius'], frame_state_vector[1]['theta'], frame_state_vector[1]['w'], frame_state_vector[1]['acceleration']])
            return frame_state_vector
        else:
            return None


    @staticmethod
    def convertToPolar(detection_object, x_center, y_center):

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

        detection_object['radius'] = radius
        detection_object['theta'] = theta
        detection_object['angular velocity'] = None
        detection_object['acceleration'] = None

        return detection_object # key/value pair object {'cls': 0, 'cnf': '0.95', 'radius': 0.0, 'theta': 0, 'w': None, 'a': None}
        #except KeyError:
        #   return detection_object

    @staticmethod
    def calcRadialDistance(curr_theta, prev_theta):  # Calculates radial distance between two points
        a = curr_theta - prev_theta
        a = (a + 180) % 360 - 180
        return a


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
