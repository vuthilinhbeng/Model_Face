
import torch
import torch.backends.cudnn as cudnn
import json
import cv2


def load_model_yolo():
    m_objD = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/lastv5chuan.pt',device='cpu')
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
    return I

I = cv2.imread("test/h187.jpg")
linh = load_model_yolo()
newimage = ObjectDetect(I,linh)
cv2.imshow("newimage",newimage)
cv2.waitKey()

cv2.imwrite("sautest/h11yy7sss3sau.jpg", newimage)
cv2.destroyAllWindows()
