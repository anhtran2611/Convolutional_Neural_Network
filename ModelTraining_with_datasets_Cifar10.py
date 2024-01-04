import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras import layers
from keras import models
from keras.utils import to_categorical #  one hot coding


(Xtrain, ytrain), (Xtest, ytest) = cifar10.load_data()
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

#Chuẩn hóa lại dữ liệu (0-255)-->(0-1)
Xtrain, Xtest = Xtrain/255, Xtest/255
#c --> One hot coding
ytrain, ytest = to_categorical(ytrain), to_categorical(ytest)




# khởi tạo mô hình dự đoán
models_training_first = models.Sequential([   # Tạo ra dạng sequential : chuỗi
    # Tăng tỉ lệ xác xuất , càng nhiều convolution thì càng nhận dạng nhiều đặc điểm
    layers.Conv2D(32,(3,3),input_shape=(32,32,3),activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0,15),


    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0,15),

    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0,15),


    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(3000, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='softmax') 
])

#models_training_first.summary() # hiển thị tổng quan về kiến trúc của mô hình.

# #Sau khi xây dựng mô hình,cần compile nó trước khi huấn luyện bằng cách chỉ định hàm mất mát, thuật toán tối ưu hóa, và các metric để theo dõi
models_training_first.compile(optimizer='adam', # bộ tối ưu hóa , áp dụng giải thuật để cập nhật lại tham số
                              loss='categorical_crossentropy', # tính toán mất mát
                               metrics=['accuracy']) # tính toán độ chính xác


# Huấn luyện mô hình
models_training_first.fit(Xtrain, ytrain, epochs=10) # 10 lần duyệt qua toàn bộ dữ liệu huấn luyện.





# # Lưu lại mô hình
# models_training_first.save('model-cifar10')
#
#
# #load model rồi dự đoán
# models = models.load_model('model-cifar10')
# pred = models.predict(Xtest[32].reshape((-1,32,32,3))) # Dự đoán hình thứ 105
# print(pred) # trả về dự đoán là 10 xác xuất sẽ rơi vào nhóm nào ? , nếu chỗ nào xác xuất lớn nhất thì hình sẽ thuộc nhóm đó
# print(classes[np.argmax(pred)]) # trả về vị trí nơi có xác xuất lớn nhất, sẽ là vị trí tên nhãn
# plt.imshow(Xtest[32])
# plt.show()
