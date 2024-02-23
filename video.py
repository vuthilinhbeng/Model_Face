
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
    #print(data_retange)
    for i in data_retange:
        dict = i
        #print("dict",dict)
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

linh = load_model_yolo()
# cv2.imshow("newimage",newimage)
# cv2.waitKey()
# vid = cv2.VideoCapture(0)
#cap=cv2.VideoCapture('test/linhome.mp4')
# while(True):
#     # Capture the video frame
#     # by frame
#     ret, frame = cap.read()
#     # Display the resulting frame
#     hi = ObjectDetect(frame,linh)
#     cv2.imshow('frame', frame)
#     # the 'q' button is set as the
#     # quitting button you may use any
#     # desired button of your choice
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# # After the loop release the cap object
# cap.release()
# # Destroy all the windows
# newimage = ObjectDetect(hi,linh)
# cv2.destroyAllWindows()()

cap = cv2.VideoCapture('test/NhanDangVuKhi/videodauvao.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Tạo đối tượng writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('videodaura.mp4', fourcc, 20, (width, height))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # xử lý video tại đây
        hi = ObjectDetect(frame,linh)
        # ghi khung hình xuống file output
        out.write(hi)
        # print("hi")
        # hiển thị video
        # cv2.imshow('frame',hi)
        # thoát video bằng phím q
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()
