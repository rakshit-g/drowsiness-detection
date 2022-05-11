# rakshit gupta imt2019516
# daksh agarwal imt2019505


import cv2 as cv
import mediapipe as mp
import time
import math
import numpy as np
import dlib
from playsound import playsound
from imutils import face_utils
from scipy.spatial import distance as dist


# variables 
frame_counter =0
CEF_COUNTER =0
TOTAL_BLINKS =0
# constants
CLOSED_EYES_FRAME =25
FONTS =cv.FONT_HERSHEY_COMPLEX

def colorBackgroundText(img, text, font, fontScale, textPos, textThickness=1,textColor=(0,255,0), bgColor=(0,0,0), pad_x=3, pad_y=3):
   
    (t_w, t_h), _= cv.getTextSize(text, font, fontScale, textThickness) # getting the text size
    x, y = textPos
    cv.rectangle(img, (x-pad_x, y+ pad_y), (x+t_w+pad_x, y-t_h-pad_y), bgColor,-1) # draw rectangle 
    cv.putText(img,text, textPos,font, fontScale, textColor,textThickness ) # draw in text

    return img


def textWithBackground(img, text, font, fontScale, textPos, textThickness=1,textColor=(0,255,0), bgColor=(0,0,0), pad_x=3, pad_y=3, bgOpacity=0.5):
  
    (t_w, t_h), _= cv.getTextSize(text, font, fontScale, textThickness) # getting the text size
    x, y = textPos
    overlay = img.copy() # coping the image
    cv.rectangle(overlay, (x-pad_x, y+ pad_y), (x+t_w+pad_x, y-t_h-pad_y), bgColor,-1) # draw rectangle 
    new_img = cv.addWeighted(overlay, bgOpacity, img, 1 - bgOpacity, 0) # overlaying the rectangle on the image.
    cv.putText(new_img,text, textPos,font, fontScale, textColor,textThickness ) # draw in text
    img = new_img

    return img

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]
 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

map_face_mesh = mp.solutions.face_mesh
# camera object 
camera = cv.VideoCapture(0)
table = np.array([((i / 255.0) ** (1.0/2)) * 255 for i in range(0, 256)]).astype("uint8")


def gamma_correction(image):
    return cv.LUT(image, table)

def histogram_equalization(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return cv.equalizeHist(gray) 
 
def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord

# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
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

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
    return ratio 

def calculate_lip(lips):
     dist1 = dist.euclidean(lips[2], lips[6]) 
     dist2 = dist.euclidean(lips[0], lips[4]) 

     LAR = float(dist1/dist2)

     return LAR


counter = 0
lip_LAR = 0.4
lip_per_frame = 20
d_blink= 0
d_ywn=0
b1=0
y1=0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

with map_face_mesh.FaceMesh(max_num_faces = 1, min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:

    # starting time here 
    start_time = time.time()
   
    while True:
        frame_counter +=1 # frame counter
        ret, frame = camera.read() 
        if not ret: 
            break 
      

        #yawn
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # faces = detector(gray)

        #blink
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_height, frame_width= frame.shape[:2]
        adjusted = gamma_correction(frame)
        adjusted = histogram_equalization(adjusted)
        rgb_frame = cv.cvtColor(adjusted, cv.COLOR_RGB2BGR)
        faces = detector(rgb_frame)
        results  = face_mesh.process(rgb_frame)

        


        if results.multi_face_landmarks:
            for (i, face) in enumerate(faces):
                mesh_coords = landmarksDetection(frame, results, False)
                ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
            
                colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2,(147,20,255), (0,255,255))

                if ratio >4:
                    CEF_COUNTER +=1
                    
                    colorBackgroundText(frame,  f'Blink', FONTS, 1.7, (int(frame_height/2), 100), 2, (0,255,255), pad_x=6, pad_y=6, )

                else:
                    if CEF_COUNTER>CLOSED_EYES_FRAME:
                        d_blink+=1
                        b1+=1
                        TOTAL_BLINKS +=1
                        CEF_COUNTER =0
            
                colorBackgroundText(frame,  f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30,150),2)
                
                cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True,(0,255,0), 1, cv.LINE_AA)
                cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, (0,255,0), 1, cv.LINE_AA)

                lips = [60,61,62,63,64,65,66,67]
                point = predictor(gray, face)
                points = face_utils.shape_to_np(point)
                lip_point = points[lips]
                LAR = calculate_lip(lip_point) 

                lip_hull = cv.convexHull(lip_point)
                cv.drawContours(frame, [lip_hull], -1, (0, 255, 0), 1)

                if LAR > lip_LAR:
                    counter += 1
                    print(counter)
                    if counter > lip_per_frame:
                        d_ywn+=1
                        y1+=1
                        cv.putText(frame, "YAWNING....", (500, 30), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                        counter=0
                        
                else:
                    counter = 0
                cv.putText(frame, "LAR: {:.2f}".format(LAR), (300, 30),cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                colorBackgroundText(frame,  f'Total yawns: {d_ywn}', FONTS, 0.7, (30,200),2)
                
                if((b1>=2 and y1 >=2) or b1>=4):
                    cv.putText(frame, "DROWSYYYYYY", (300, 100), cv.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 2)
                    playsound("alarm.wav")
                    time.sleep(3)
                    # cv.putText(frame, " ", (300, 100), cv.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 2)
                    if(b1>=4): b1=b1%4
                    y1=0

        else:
            colorBackgroundText(frame,  f'Unable to detect face', FONTS, 1, (int(frame_height/2), 100), 2, (0,255,255), pad_x=6, pad_y=6, )
            
        # calculating  frame per seconds FPS
        end_time = time.time()-start_time
        fps = frame_counter/end_time

        frame =textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
    
        cv.imshow('frame', frame)
        key = cv.waitKey(2)
        if key==ord('q') or key ==ord('Q'):
            break
    cv.destroyAllWindows()
    camera.release()