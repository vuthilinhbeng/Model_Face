import torch
import torch.backends.cudnn as cudnn
import json
import cv2

def load_model_yolo():
    m_objD = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/last_v5_16_100.pt',device='cpu')
    return m_objD#  hàm nhận diện đối tượng yolo

def ObjectDetect(I,m_objD):
    results = m_objD(I)
    data = results.pandas().xyxy[0].to_json(orient="records")
    data_retange = json.loads(data)
    list_text={}
    print(data_retange)
    for i in data_retange:
        dict = i
        print("dict",dict)
        xmin = int(dict['xmin'])
        xmax = int(dict['xmax'])
        ymin = int(dict['ymin'])
        ymax = int(dict['ymax'])
        classs = int(dict['class'])
        name = i["name"]
        confidence = str(i["confidence"])
        
        I_crop = I[ymin:ymax, xmin:xmax]
        # cv2.imshow("hiha",I_crop)
        start_point = (xmin, ymin)
        end_point = (xmax, ymax)
        color = (0, 255, 0)
        color2 = (255,	255	,0)
        thickness = 2
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale
        fontScale = 1
        # viet name
        if(classs == 1):
            cv2.putText(I, name, start_point, font, 
                   fontScale, color2, thickness, cv2.LINE_AA)
            cv2.putText(I, confidence, end_point, font, 
                   fontScale, color2, thickness, cv2.LINE_AA)
            cv2.rectangle(I, start_point, end_point, color2, thickness)
        else:
            cv2.putText(I, name, start_point, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(I, confidence, end_point, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
            cv2.rectangle(I, start_point, end_point, color, thickness)
 
        
        # ve toa do
        # cv2.rectangle(I, start_point, end_point, color, thickness)
        # cv2.imshow("hihi",I)
        # cv2.waitKey()

    return I
linh = load_model_yolo()
# cv2.imshow("newimage",newimage)
# cv2.waitKey()
vid = cv2.VideoCapture(0)
# cap=cv2.VideoCapture('video3.mp4')
while(True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    # Display the resulting frame
    hi = ObjectDetect(frame,linh)
    cv2.imshow('frame', frame)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# After the loop release the cap object
vid.release()
# Destroy all the windows
newimage = ObjectDetect(hi,linh)
cv2.destroyAllWindows()
