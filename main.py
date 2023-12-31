import cv2 as cv
import mediapipe as mp
import time
import utils, math
import numpy as np

# variables
frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0
# constants
CLOSED_EYES_FRAME = 3
FONTS = cv.FONT_HERSHEY_COMPLEX

# face bounder indices 
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# lips indices for Landmarks
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
        37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
# Left eyes indices 
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

map_face_mesh = mp.solutions.face_mesh

# camera object
camera = cv.VideoCapture(0)


# landmark detection function
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]
    lips_coords = [mesh_coord[i] for i in LIPS]
    cv.polylines(img, [np.array(lips_coords, dtype=np.int32)], True, utils.RED, 1, cv.LINE_AA)

    # returning the list of tuples for each landmarks
    return mesh_coord


# Eyebrow detector
def draw_eyebrows(img, landmarks, eyebrow_indices):
    for i in range(len(eyebrow_indices) - 1):
        cv.line(img, landmarks[eyebrow_indices[i]], landmarks[eyebrow_indices[i + 1]], utils.YELLOW, 2)


# Euclaidean distance
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance


# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    ratio = (reRatio + leRatio) / 2
    return ratio


# Eyes Extrctor function,
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # converting color image to  scale image 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # getting the dimension of image 
    dim = gray.shape

    # creating mask from gray scale dim
    mask = np.zeros(dim, dtype=np.uint8)

    # drawing Eyes Shape on mask with white color 
    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # showing the mask 
    # cv.imshow('mask', mask)

    # draw eyes image on mask, where white shape is 
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    # change black color to gray other than eys 
    # cv.imshow('eyes draw', eyes)
    eyes[mask == 0] = 155

    # getting minium and maximum x and y  for right and left eyes 
    # For Right Eye 
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item: item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # For LEFT Eye
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item: item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # croping the eyes from mask 
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    # returning the cropped eyes 
    return cropped_right, cropped_left


# Eyes Postion Estimator
def positionEstimator(cropped_eye):
    # getting height and width of eye 
    h, w = cropped_eye.shape

    # remove the noise from images
    gaussain_blur = cv.GaussianBlur(cropped_eye, (9, 9), 0)
    median_blur = cv.medianBlur(gaussain_blur, 3)

    # applying thrsholding to convert binary_image
    ret, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)

    # create fixd part for eye with 
    piece = int(w / 3)

    # slicing the eyes into three parts 
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece + piece]
    left_piece = threshed_eye[0:h, piece + piece:w]

    # calling pixel counter function
    eye_position, color = pixelCounter(right_piece, center_piece, left_piece)

    return eye_position, color


# creating pixel counter function
def pixelCounter(first_piece, second_piece, third_piece):
    # counting black pixel in each part 
    right_part = np.sum(first_piece == 0)
    center_part = np.sum(second_piece == 0)
    left_part = np.sum(third_piece == 0)
    # creating list of these values
    eye_parts = [right_part, center_part, left_part]

    # getting the index of max values in the list 
    max_index = eye_parts.index(max(eye_parts))
    pos_eye = ''
    if max_index == 0:
        pos_eye = "RIGHT"
        color = [utils.BLACK, utils.GREEN]
    elif max_index == 1:
        pos_eye = 'CENTER'
        color = [utils.YELLOW, utils.PINK]
    elif max_index == 2:
        pos_eye = 'LEFT'
        color = [utils.GRAY, utils.YELLOW]
    else:
        pos_eye = "Closed"
        color = [utils.GRAY, utils.YELLOW]
    return pos_eye, color


def detect_emotion(eye_coords, eyebrow_coords, lips_coords):
    # Calculate eyebrow position
    eyebrow_mean_y = np.mean([point[1] for point in eyebrow_coords])

    # Calculate eye aspect ratio for eye openness
    eye_aspect_ratio = (euclaideanDistance(eye_coords[1], eye_coords[5]) +
                        euclaideanDistance(eye_coords[2], eye_coords[4])) / (
                               2 * euclaideanDistance(eye_coords[0], eye_coords[3]))

    # Calculate mouth aspect ratio for mouth openness
    mouth_aspect_ratio = (euclaideanDistance(lips_coords[14], lips_coords[18]) +
                          euclaideanDistance(lips_coords[12], lips_coords[16])) / (
                                 2 * euclaideanDistance(lips_coords[2], lips_coords[6]))

    # Define thresholds for eye and mouth position to determine emotions
    if mouth_aspect_ratio > 0.5 and eyebrow_mean_y < 120:
        return "Surprised"
    elif eye_aspect_ratio > 0.2 and eyebrow_mean_y > 150:
        return "Neutral"
    elif mouth_aspect_ratio < 0.2 and eyebrow_mean_y > 200:
        return "Sad"
    else:
        return "Angry"


def stress_blinks(total_blinks, recording_time):
    if total_blinks>recording_time :
        return 1.0
    else:
        return 0.1


def stress_eye_position(right_eye, left_eye):
    if (right_eye == "RIGHT" and left_eye == "RIGHT") or (right_eye == "LEFT" and left_eye == "LEFT"):
        return 1.0
    else:
        return 0.0


def stress_emotion(emotion):
    if emotion == "Surprised":
        return 1.0
    elif (emotion == "Sad") or (emotion == "Angry"):
        return 0.8
    else:
        return 0.3


# Function to calc
with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    # starting time here
    start_time = time.time()
    # starting Video loop here.
    while True:
        frame_counter += 1  # frame counter
        ret, frame = camera.read()  # getting frame from camera
        if not ret:
            break  # no more frames break
        #  resizing frame

        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
            utils.colorBackgroundText(frame, f'Ratio : {round(ratio, 2)}', FONTS, 0.7, (30, 100), 2, utils.PINK,
                                      utils.YELLOW)

            if ratio > 3.5:
                CEF_COUNTER += 1
                utils.colorBackgroundText(frame, f'Blink', FONTS, 1.7, (int(frame_height / 2), 100), 2, utils.YELLOW,
                                          pad_x=6, pad_y=6, )

            else:
                if CEF_COUNTER > CLOSED_EYES_FRAME:
                    TOTAL_BLINKS += 1
                    CEF_COUNTER = 0
            utils.colorBackgroundText(frame, f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30, 150), 2)

            cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, utils.GREEN, 1,
                         cv.LINE_AA)
            cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, utils.GREEN, 1,
                         cv.LINE_AA)

            # Blink Detector Counter Completed
            right_coords = [mesh_coords[p] for p in RIGHT_EYE]
            # print(right_coords)
            left_coords = [mesh_coords[p] for p in LEFT_EYE]
            # print(left_coords)
            crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords)
            # cv.imshow('right', crop_right)
            # cv.imshow('left', crop_left)
            eye_position, color = positionEstimator(crop_right)
            utils.colorBackgroundText(frame, f'R: {eye_position}', FONTS, 1.0, (40, 220), 2, color[0], color[1], 8, 8)
            eye_position_left, color = positionEstimator(crop_left)
            utils.colorBackgroundText(frame, f'L: {eye_position_left}', FONTS, 1.0, (40, 320), 2, color[0], color[1], 8,
                                      8)
            lips_coords = [mesh_coords[p] for p in LIPS]

            # Drawing lips contour on the frame
            cv.polylines(frame, [np.array(lips_coords, dtype=np.int32)], True, utils.RED, 1, cv.LINE_AA)

            # Drawing eyebrows contour on the frame
            draw_eyebrows(frame, mesh_coords, LEFT_EYEBROW)
            draw_eyebrows(frame, mesh_coords, RIGHT_EYEBROW)

            # Detect emotion
            emotion = detect_emotion(
                [mesh_coords[p] for p in LEFT_EYEBROW + RIGHT_EYEBROW],
                [mesh_coords[p] for p in LEFT_EYE],
                lips_coords
            )
            utils.colorBackgroundText(frame, f'Emotion: {emotion}', FONTS, 0.7, (30, 200), 2)
            # stress_ratio = 0.3 * stress_blinks(TOTAL_BLINKS, time.time() - start_time) + 0.3 * stress_emotion(
            #     emotion) + 0.3 * stress_eye_position(eye_position, eye_position_left)
            # frame = utils.textWithBackground(frame, f'stress ration: {round(stress_ratio, 1)}', FONTS, 1.0, (30, 80),
            #                                  bgOpacity=0.9,
            #                                  textThickness=2)

        # calculating  frame per seconds FPS
        end_time = time.time() - start_time

        fps = frame_counter / end_time

        frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)}', FONTS, 1.0, (30, 50), bgOpacity=0.9,
                                         textThickness=2)

        stress_ratio = 0.3 * stress_blinks(TOTAL_BLINKS, time.time() - start_time) + 0.3 * stress_emotion(
            emotion) + 0.3 * stress_eye_position(eye_position, eye_position_left)
        frame = utils.textWithBackground(frame, f'Stress ratio: {round(stress_ratio, 1)}', FONTS, 1.0, (30, 80),
                                         bgOpacity=0.9,
                                         textThickness=2)

        cv.imshow('frame', frame)
        key = cv.waitKey(2)
        if key == ord('q') or key == ord('Q'):
            break
    cv.destroyAllWindows()
    camera.release()
