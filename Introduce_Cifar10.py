import numpy as np
import matplotlib.pyplot as plt
# flat dữ liệu bằng keras với tensorflow
#cifar10 gôm 50 nghìn tấm hình màu, kích cỡ 32x32
from keras.datasets import cifar10


# download bộ ảnh cifar10
(Xtrain, ytrain), (Xtest, ytest) = cifar10.load_data()
# X : giá trị tấm hình , Xtrain : tập dữ liệu huấn luyện, Xtest : tập dữ liệu kiểm thử
#y : giá trị label, giám sát, biết trước label, VD : mèo, chó
# với mỗi tấm hình X sẽ có nhãn tương ứng là y
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
#print(ytrain[0][0]) # ìm ra nhãn của tấm hình đầu tiên=>>in ra nhãn là ở vị trí số [6]==> nhãn là 'frog'(0-6)

# print(Xtrain.shape, Xtest.shape) #(50000 ,32,32,3) 50000 tấm , 32x32, 3 layer ; test 10000 tấm


#Xem 50 tấm hình baats kì
for i in range (50):
    plt.subplot(5,10,i+1)
    plt.imshow(Xtrain[i])
    plt.title(classes[ytrain[i][0]]) #[0] đều là giá trị nhãn của một ảnh cụ thể trong tập huấn luyện.
    plt.axis('off')

plt.show()

# Mục đích :xây dựng 1 model dựa trên 50 nghìn tấm hình , sau khi có model sẽ đến bước train, sau khi train xong sẽ đưa một số lượng hình bất kì thì máy sẽ phải đoán đó là hình thuộc nhóm nào
