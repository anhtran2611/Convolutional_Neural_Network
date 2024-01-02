# giả sử ảnh truyền vào có kích cỡ mxm, cornel có kích cỡ 3x3 ---> size của kết quả sẽ là (m-3+1, m-3+1)
import cProfile
import cv2
import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(20)
img = cv2.imread('D:/Image_database/Tenth_Doctor2.jpg')
# print(img.shape) xem kích cơ ảnh

# resize
img = cv2.resize(img, (200, 200))
# Chuyển ảnh qua màu xám , nếu k chuyển ảnh ra màu xám , ảnh gốc sẽ là ảnh 3 chieu (200,200,3) voi 3 layer
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Tạo hàm đưa ảnh và kernel vào

def convolution2d(input, kernelSize):
    height, width = input.shape  # height ,width : 200
    kernel = np.random.randn(kernelSize, kernelSize) #randn để tạo số ngẫu nhiên
    # print(kernel)

    #2.Tạo biến kết quả để hứng
    results = np.zeros((height - kernelSize + 1, width - kernelSize + 1)) # np.zeros để tạo được 1 mảng có kích cỡ tương ứng với chiều rộng , chiều cao truyền vào
   # print(results, results.shape)

    # 1.tạo vòng lặp để kernel chiếu vào ảnh input theo thứ tự , nó sẽ đi từng nấc một

    for row in range(0,height - kernelSize + 1): # chỉ dịch chuyển được height - kernelSize + 1 vị trí
      for column in range(0, width - kernelSize + 1):
        results[row,column] = np.sum(input[row: row + kernelSize, column: column +kernelSize] * kernel)   #Duyệt chính xác theo kích cỡ của kernel để đối chiếu ô vào ảnh input, sau đó nhân với kernel rồi lấy tổng

    return results


#convolution2d(img_gray,3)  # gọi hàm in ra mảng 3x3


# hiển thị ra hình ảnh
plt.imshow(convolution2d(img_gray,3))
plt.show()
