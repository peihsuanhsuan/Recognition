import cv2
from deepface import DeepFace
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf

# 定義該情緒的中文字
text_obj={
    'angry': '生氣',
    'disgust': '噁心',
    'fear': '害怕',
    'happy': '開心',
    'sad': '難過',
    'surprise': '驚訝',
    'neutral': '正常'
}

# 定義加入文字函式
def putText(x,y,text,size=50,color=(255,255,255)):
    global img
    fontpath = 'NotoSansTC-Regular.otf'            # 字型
    font = ImageFont.truetype(fontpath, size)      # 定義字型與文字大小
    imgPil = Image.fromarray(img)                  # 轉換成 PIL 影像物件
    draw = ImageDraw.Draw(imgPil)                  # 定義繪圖物件
    draw.text((x, y), text_obj[text], fill=color, font=font) # 加入文字
    img = np.array(imgPil)                         # 轉換成 np.array

model = tf.keras.models.load_model('keras_model.h5', compile=False)  # 載入模型
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)          # 設定資料陣列

def text(text):      # 建立顯示文字的函式
    global img       # 設定 img 為全域變數
    org = (0,50)     # 文字位置
    fontFace = cv2.FONT_HERSHEY_SIMPLEX  # 文字字型
    fontScale = 1                        # 文字尺寸
    color = (255,255,255)                # 顏色
    thickness = 2                        # 文字外框線條粗細
    lineType = cv2.LINE_AA               # 外框線條樣式
    cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType) # 放入文字

recognizer = cv2.face.LBPHFaceRecognizer_create()         # 啟用訓練人臉模型方法
recognizer.read('face.yml')                               # 讀取人臉模型檔
cascade_path = "haarcascade_frontalface_default.xml"  # 載入人臉追蹤模型
face_cascade = cv2.CascadeClassifier(cascade_path)        # 啟用人臉追蹤

cap = cv2.VideoCapture(0)                                 # 開啟攝影機
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, img = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    img = cv2.resize(img,(540,300))              # 縮小尺寸，加快辨識效率
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # 轉換成黑白
    faces = face_cascade.detectMultiScale(gray)  # 追蹤人臉 ( 目的在於標記出外框 )

    # 建立姓名和 id 的對照表
    name = {
        '1':'nine',
        '2':'Tzuyu'
    }

    img2 = cv2.resize(img , (398, 224))
    img2 = img2[0:224, 80:304]
    image_array = np.asarray(img2)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    a,b,bg= prediction[0]    # 印出每個項目的數值資訊
    print(a,b,bg)
    if a>0.999:              # 判斷有戴口罩
        text('ok~')
    if b>0.001:              # 判斷沒戴口罩
        text('no mask!!')
    cv2.imshow('oxxostudio', img) 

    # 依序判斷每張臉屬於哪個 id
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)            # 標記人臉外框
        idnum,confidence = recognizer.predict(gray[y:y+h,x:x+w])  # 取出 id 號碼以及信心指數 confidence
        print(confidence)
        if confidence < 80:
            text = name[str(idnum)]                               # 如果信心指數小於 60，取得對應的名字
            print(text)
        else:
            text = 'Who are you?'                                          # 不然名字就是 ???
        # 在人臉外框旁加上名字
        cv2.putText(img, text, (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        try:
            # analyze = DeepFace.analyze(img, actions=['emotion'])
            emotion = DeepFace.analyze(img, actions=['emotion'])[0]['dominant_emotion']  # 取得情緒文字
            putText(0,40,emotion)                  # 放入文字
        except:
            pass

                                    
    if cv2.waitKey(5) == ord('q'):
        break    # 按下 q 鍵停止
cap.release()
cv2.destroyAllWindows()