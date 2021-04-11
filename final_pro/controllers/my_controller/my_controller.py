from controller import Robot, Motor, Gyro, GPS, Camera, Compass, Keyboard, LED, InertialUnit, DistanceSensor
import math
import cv2
import numpy as np
import time
from PIL import Image
import pytesseract as tess
tess.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/4.1.1/bin/tesseract'

strt_tim = time.time()


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def biggest_cont(mask):
    contour, her = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_cont = 0
    for i in range(len(contour)):
        if cv2.contourArea(contour[i]) >= max_cont:
            final = contour[i]
            max_cont = cv2.contourArea(contour[i])
    return final


def align_img(img):
    def mask_white(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lowwhite = np.array([0, 0, 200])
        upwhite = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lowwhite, upwhite)
        return mask

    mask = mask_white(img)
    final = biggest_cont(mask)

    # while len(final) != 4:
    epsilon = 0.02 * cv2.arcLength(final, True)
    final = cv2.approxPolyDP(final, epsilon, True)

    square_points = np.squeeze(final)
    # cv2.drawContours(black, [final], -1, (0, 255, 0), 1)

    s1 = square_points[0] - square_points[1]
    s2 = square_points[1] - square_points[2]
    if (pow(s1[0], 2) + pow(s1[1], 2)) < (pow(s2[0], 2) + pow(s2[1], 2)):
        tan = s2[1] / s2[0]
    else:
        tan = s1[1] / s1[0]

    rot_angle = math.atan(tan) * 57.2958
    rotated_img = rotate_image(img, rot_angle)
    mask = mask_white(rotated_img)
    finall = biggest_cont(mask)

    # while len(finall) != 4:
    epsilon = 0.01 * cv2.arcLength(finall, True)
    finall = cv2.approxPolyDP(finall, epsilon, True)
    # if p==0:
    #     align_img(rotated_img,p=1)#recussion

    x, y, w, h = cv2.boundingRect(finall)
    cropped = rotated_img[y:y + h, x:x + w]

    return cropped


row = [-1, -1, -1, 0, 1, 0, 1, 1]
col = [-1, 1, 0, -1, -1, 1, 0, 1]


def isSafe(x, y, processed):
    M = 3
    N = 3
    return (0 <= x < M) and (0 <= y < N) and not processed[x][y]


def searchBoggle(board, words, processed, i, j, path=""):

    processed[i][j] = True
    path = path + board[i][j]
    words.add(path)
    for k in range(8):
        if isSafe(i + row[k], j + col[k], processed):
            searchBoggle(board, words, processed, i + row[k], j + col[k], path)
    processed[i][j] = False


def searchInBoggle(board, input):
    M = 3
    N = 3
    processed = [[False for x in range(N)] for y in range(M)]
    words = set()
    for i in range(M):
        for j in range(N):
            searchBoggle(board, words, processed, i, j)

    # for each word in the input list, check whether it is present in the set
    return ([word for word in input if word in words])


def landing_coordinates(words, mat):
    wrd = []
    for t in words:
        wrd.append(t[0])
    k = np.array(mat)
    k = np.reshape(k, (3, 3))
    mat = k.tolist()
    words_found = searchInBoggle(mat, wrd)
    num = wrd.index(words_found[0])
    return [words[num][1], words[num][2]]


def get_qr_letter(aligned_img):
    lhsv = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2HSV)
    maskgreen = cv2.inRange(lhsv, np.array(
        [36, 50, 70]), np.array([89, 255, 255]))
    maskblue = cv2.inRange(lhsv, np.array(
        [85, 30, 30]), np.array([130, 255, 255]))
    cont_green = biggest_cont(maskgreen)
    cont_blue = biggest_cont(maskblue)

    if cont_blue[0, 0, 0] < cont_green[0, 0, 0]:  # right image no change to be made
        ans = aligned_img
    else:
        ans = np.fliplr(aligned_img)
        ans = np.flipud(ans)
    division = len(ans[0]) // 2
    qr_code = ans[:, :(division + 2)]
    letter = ans[:, division:]
    return qr_code, letter


def maintaing_altitude(h, c=False):
    final_position = drone.gps.getValues()[1] + h
    drone.target_altitude = final_position
    drone.robot.step(drone.timestep)
    if c:
        i = 0
        while True:
            i += 1
            drone.move('up', 0)  # move down
            if i > 300:
                break


def convert_to_list(string):
    x = string.find(',')
    c1 = string[1:x]
    c2 = string[(x + 1):string.find(')')]
    return [float(c1), float(c2)]


def qr_data(qr_img):
    qrDecoder = cv2.QRCodeDetector()
    # Detect and decode the qrcode
    data, bbox, rectifiedImage = qrDecoder.detectAndDecode(qr_img)
    return data


def let_recog(img):
    # Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((2, 3), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    gray = cv2.erode(gray, kernel, iterations=2)
    _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite("thres.png", gray)

    y = tess.image_to_string(Image.open('thres.png'), config='--psm 10')

    if y[0] == '|':
        # print("aaaaaa")
        return 'I'

    return y[0]


def procces_img(img, check=False, eighth_building=False):  # kkkkkkkkkkkkkkkkkkkk
    if check:
        rotated = align_img(img)
        qrcode, letter = get_qr_letter(rotated)
        if eighth_building:
            temporary = np.flipud(letter)
            temporary = np.fliplr(temporary)
            letter = np.flipud(qrcode)
            letter = np.fliplr(letter)
            qrcode = temporary

        qr_data1 = qr_data(qrcode)
        if len(qr_data1) == 0:

            maintaing_altitude(-3, True)

            return "decend", 3
        else:
            dimmmm = letter.shape
            k = letter[int(0.2 * dimmmm[0]): int(0.8 * dimmmm[0]),
                       : int(0.77 * dimmmm[1])]

            letter = let_recog(k)
            return qr_data1, letter

    else:
        rotated = align_img(img)
        qrcode, letter = get_qr_letter(rotated)
        qr_data1 = qr_data(qrcode)

        return qr_data1, letter


def moveup_fs():  # disturbance in front sensor
    drone.move('up', 1, True)

    drone.robot.step(drone.timestep)
    drone.target_altitude = drone.gps.getValues()[1] + 3


def final_building(image):
    img = align_img(image)
    dim = img.shape
    x = dim[0] // 2
    y = dim[1] // 2
    image1 = img[:x, :y]
    image2 = img[x:, :y]
    image3 = img[:x, y:]
    image4 = img[x:, y:]
    a, b, c, d = qr_data(image1), qr_data(
        image2), qr_data(image3), qr_data(image4)
    if len(a) == 0 or len(b) == 0 or len(c) == 0 or len(d) == 0:
        maintaing_altitude(-3, True)

        return "decend"

    else:
        d1 = [a, b, c, d]
        data = []
        for i in d1:
            comma_1 = i.find(',')
            comma_2 = i.find(',', comma_1 + 1)
            end = i.find(')')
            part_1 = i[1:comma_1]
            part_2 = float(i[comma_1 + 1: comma_2])
            part_3 = float(i[comma_2 + 1: end])
            f = [part_1, part_2, part_3]
            data.append(f)

        return data


def land_it(coordinates):
    quit = 0
    rk = 0
    shutter = 0
    while drone.robot.step(drone.timestep) != -1:

        vec = np.array([drone.compass.getValues()[0],
                        drone.compass.getValues()[1]])
        mag = np.linalg.norm(vec)
        sin_ang = math.asin(vec[1] / mag) * 57.2958
        cos_ang = math.acos(vec[0] / mag) * 57.2958

        if cos_ang < 2:

            differ = [coordinates[0] - (drone.gps.getValues()[0]),
                      coordinates[1] - (drone.gps.getValues()[2])]
            if sin_ang > 0:
                drone.move('left', 0.005)
                drone.robot.step(drone.timestep)
            if sin_ang < 0:
                drone.move('right', 0.005)
                drone.robot.step(drone.timestep)

            if abs(differ[0]) > 0.05:
                p = differ[0] / abs(differ[0])
                drone.move('forward', 0.5 * p)
                drone.robot.step(drone.timestep)

            if abs(differ[1]) > 0.05:
                r = differ[1] / abs(differ[1])
                drone.move('sLeft', -0.5 * r)
                drone.robot.step(drone.timestep)

        if cos_ang > 2 and sin_ang > 0:

            if cos_ang > 18:
                drone.move('left', 0.1)
                drone.robot.step(drone.timestep)

            else:
                drone.move('left', 0.02)
                drone.robot.step(drone.timestep)
            # print("left")
        if cos_ang > 2 and sin_ang < 0:

            if cos_ang > 18:
                drone.move('right', 0.1)
                drone.robot.step(drone.timestep)
            else:
                drone.move('right', 0.02)
                drone.robot.step(drone.timestep)

        if drone.ds_bottom.getValue() > 1900 or shutter == 1:
            shutter = 1

            if cos_ang < 2:

                if abs(differ[0]) < 0.05 and abs(differ[1]) < 0.05:
                    if rk == 0:
                        for i in range(330):

                            if i < 200:
                                drone.move('down', 0.002)
                                drone.robot.step(drone.timestep)
                            else:
                                drone.move('down', 0)
                                drone.robot.step(drone.timestep)

                    rk += 1
                    if rk == 1:

                        drone.front_left_motor.setVelocity(0)
                        drone.front_right_motor.setVelocity(0)
                        drone.rear_left_motor.setVelocity(0)
                        drone.rear_right_motor.setVelocity(0)
                        drone.robot.step(drone.timestep)
                        quit = 1

        elif shutter == 0:
            if drone.ds_bottom.getValue() < 1000:
                drone.target_altitude = drone.gps.getValues()[1] - 1

                drone.move('down', 0)

            else:
                drone.target_altitude = drone.gps.getValues()[1]

                drone.move('down', 0)

        if quit == 1:
            break
    return quit


def move_to_dest(i):
    global done
    i = i - 1

    print(all_letters)

    while drone.gps.getValues()[0] >= (coordinates[i][0] + 0.5) or drone.gps.getValues()[0] <= (coordinates[i][0] - 0.5) or drone.gps.getValues()[2] <= (coordinates[i][1] - 0.5) or drone.gps.getValues()[2] >= (coordinates[i][1] + 0.5):

        path_vec = np.array([coordinates[i][0] - drone.gps.getValues()[0],
                             coordinates[i][1] - drone.gps.getValues()[2]])
        mag_pv = np.linalg.norm(path_vec)
        drone_vec = np.array(
            [drone.compass.getValues()[0], drone.compass.getValues()[1]])
        mag_dv = np.linalg.norm(drone_vec)
        dot_pro = path_vec @ drone_vec  # dot product
        # diffrence angele by cos inverse
        angle_c = math.acos((dot_pro) / (mag_dv * mag_pv)) * 57.2958
        cross = np.cross(path_vec, drone_vec)  # cross product
        angle_s = math.asin((cross) / (mag_dv * mag_pv)) * \
            57.2958         # diffrence angle by sin inverse

        if drone.ds_front.getValue() > 50:
            moveup_fs()

        if angle_c <= 5:

            if angle_s > 0:
                drone.move('left', 0.005)
            elif angle_s < 0:
                drone.move('right', 0.005)
            # if mag_pv < 13:
                # drone.move('forward', 1)

            drone.move('forward', 2)

        elif drone.ds_bottom.getValue() > 5 and mag_pv < 25:
            h = 12 - ((drone.ds_bottom.getValue() * 12) / 2000)
            h = 28 - h
            maintaing_altitude(h)
        elif angle_c > 5 and angle_s > 0:

            if angle_c > 15:
                drone.move('left', 0.2)
            else:
                drone.move('left', 0.05)
            # print("left")
        elif angle_c > 5 and angle_s < 0:

            if angle_c > 15:
                drone.move('right', 0.2)
            else:
                drone.move('right', 0.05)

        drone.robot.step(drone.timestep)

    drone.camera.saveImage("hello.png", 25)
    img = cv2.imread("hello.png")
    if len(coordinates) == 10:
        final_qrs = final_building(img)

        if final_qrs != "decend":
            land_here = landing_coordinates(final_qrs, all_letters)
            print("Landing coordinates --> ", end=" ")
            print(land_here)
            done = land_it(land_here)

    else:
        # exception case because the border colour of qr code at 8th building is not blue its green
        if len(coordinates) == 8:
            qr1, let1 = procces_img(img, True, True)
        else:
            qr1, let1 = procces_img(img, True)

        if qr1 != "decend":
            qrcode1 = convert_to_list(qr1)
            coordinates.append(qrcode1)
            all_letters.append(let1)


def SIGN(x): return int(x > 0) - int(x < 0)
def CLAMP(value, low, high): return min(high, max(value, low))


class Drone:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        self.camera = self.robot.getDevice('camera')
        self.camera.enable(self.timestep)
        # front_left_led = robot.getDevice("front left led");
        # front_right_led = robot.getDevice("front right led");
        self.imu = self.robot.getDevice("inertial unit")
        self.imu.enable(self.timestep)
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.timestep)
        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.timestep)
        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(self.timestep)
        self.ds_front = self.robot.getDevice("ds_front")
        self.ds_front.enable(self.timestep)
        self.ds_right = self.robot.getDevice("ds_right")
        self.ds_right.enable(self.timestep)
        self.ds_left = self.robot.getDevice("ds_left")
        self.ds_left.enable(self.timestep)
        self.ds_bottom = self.robot.getDevice("ds_bottom")
        self.ds_bottom.enable(self.timestep)

        # keyboard = Keyboard();
        # keyboard.enable(timestep)
        self.camera_roll_motor = self.robot.getDevice('camera roll')
        self.camera_pitch_motor = self.robot.getDevice('camera pitch')

        self.front_left_motor = self.robot.getDevice("front left propeller")
        self.front_right_motor = self.robot.getDevice("front right propeller")
        self.rear_left_motor = self.robot.getDevice("rear left propeller")
        self.rear_right_motor = self.robot.getDevice("rear right propeller")
        self.motors = [self.front_left_motor, self.front_right_motor,
                       self.rear_left_motor, self.rear_right_motor]

        for motor in self.motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(1.0)

        self.k_vertical_thrust = 68.5
        self.k_vertical_offset = 0.6
        self.k_vertical_p = 3.0
        self.k_roll_p = 50.0
        self.k_pitch_p = 30.0

        self.target_altitude = 1.0

    def move(self, command, intensity, temp=False):
        roll = self.imu.getRollPitchYaw()[0] + math.pi / 2.0
        pitch = self.imu.getRollPitchYaw()[1]
        altitude = self.gps.getValues()[1]
        roll_acceleration = self.gyro.getValues()[0]
        pitch_acceleration = self.gyro.getValues()[1]

        # led_state = int(time) % 2
        # front_left_led.set(led_state)
        # front_right_led.set(int(not led_state))

        self.camera_roll_motor.setPosition(-0.115 * roll_acceleration)
        self.camera_pitch_motor.setPosition(-0.1 * pitch_acceleration)

        roll_disturbance = 0.0
        pitch_disturbance = 0.0
        yaw_disturbance = 0.0

        if(command == 'forward'):
            pitch_disturbance = intensity  # 2.0
        elif(command == 'backward'):
            pitch_disturbance = -intensity  # -2.0
        elif(command == 'right'):
            yaw_disturbance = intensity  # 1.3
        elif(command == 'left'):
            yaw_disturbance = -intensity  # -1.3
        elif(command == 'sRight'):
            roll_disturbance = intensity  # -1.0
        elif(command == 'sLeft'):
            roll_disturbance = intensity  # 1.0
        elif(command == 'up'):
            if temp:
                self.tem_alt = self.target_altitude
            self.target_altitude += intensity  # 0.05
        elif(command == 'down'):
            if temp:
                self.tem_alt = self.target_altitude
            self.target_altitude -= intensity  # 0.05

        roll_input = self.k_roll_p * \
            CLAMP(roll, -1.0, 1.0) + roll_acceleration + roll_disturbance
        pitch_input = self.k_pitch_p * \
            CLAMP(pitch, -1.0, 1.0) - pitch_acceleration + pitch_disturbance
        yaw_input = yaw_disturbance
        clamped_difference_altitude = CLAMP(
            self.target_altitude - altitude + self.k_vertical_offset, -1.0, 1.0)
        vertical_input = self.k_vertical_p * \
            pow(clamped_difference_altitude, 3.0)

        front_left_motor_input = self.k_vertical_thrust + \
            vertical_input - roll_input - pitch_input + yaw_input
        front_right_motor_input = self.k_vertical_thrust + \
            vertical_input + roll_input - pitch_input - yaw_input
        rear_left_motor_input = self.k_vertical_thrust + \
            vertical_input - roll_input + pitch_input - yaw_input
        rear_right_motor_input = self.k_vertical_thrust + \
            vertical_input + roll_input + pitch_input + yaw_input
        self.front_left_motor.setVelocity(front_left_motor_input)
        self.front_right_motor.setVelocity(-front_right_motor_input)
        self.rear_left_motor.setVelocity(-rear_left_motor_input)
        self.rear_right_motor.setVelocity(rear_right_motor_input)
        if temp:
            self.target_altitude = self.tem_alt
        # print(yaw_disturbance)


drone = Drone()
coordinates = []
all_letters = []


i = 0
p = 0
j = 0
done = 0

while drone.robot.step(drone.timestep) != -1:
    j += 1
    if i == 0:
        drone.move('up', 34)
        i = i + 1

    drone.move('up', 0)
    if drone.gps.getValues()[1] > 35:  # for first
        if p == 0:
            drone.camera.saveImage("hello.png", 25)
            img = cv2.imread("hello.png")
            qr_img = align_img(img)
            data = convert_to_list(qr_data(qr_img))
            coordinates.append(data)

            p += 1

    if len(coordinates) > 0:
        move_to_dest(len(coordinates))
    if done == 1:
        break
print(".")
# tim = time.time() - strt_tim
print("time taken  ---> ", str(drone.robot.getTime()), " seconds")
print(".")
print("----------Task completed by Team solaris-----------")
