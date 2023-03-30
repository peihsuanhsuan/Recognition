TrainingData
1. 事先訓練模型，分成兩部份，先訓練2組人臉照片(馬英九及周子瑜)，訓練出face.yml。
2. 用Teachable Machine 將戴口罩及沒戴口罩各50張照片、不同背景照片18張訓練模型 keras_model.h5 檔案。
3. 除了訓練此兩個模型外，還有套用網路上通用的haarcascade_frontalface_default.xml。

補充：NotoSansTC-Regular.otf 為字體之檔案，可忽略

Result:
1. 未戴口罩時，會偵測到情緒並顯示於視窗左側
2. 人臉範圍會以綠色方框標註並在方框上方顯示文字，分為人名(或是Who are you?) + 是否佩戴口罩(有配戴者會顯示”ok~”；未配戴者會顯示” no mask!!”)