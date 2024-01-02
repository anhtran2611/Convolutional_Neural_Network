#rol = region of interesting : vùng diện tích vào kernel chiếu vào
# có input lấy shape[0] là có row , shape[1] là có column nên k cần height và width nữa , tương tự với kernelSize
#Stride là bước nhảy khi chập trên ảnh , bước nhảy càng nhiều thông tin càng mất nhiều
# 16 Kernel thì nhân với input được 16 kết quả
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = cv2.imread('D:/Image_database/Tenth_Doctor2.jpg')
# resize
img = cv2.resize(img, (200, 200))
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

class convolution2d:
    def __init__(self, input, numOfKernel=8, kernelSize=3, padding=0, stride=1):
        self.input = np.pad(input,((padding,padding),(padding,padding)),'constant')          # Gắn padding vào input
        self.stride = stride
        self.kernel = np.random.randn(numOfKernel, kernelSize, kernelSize) # in ra số layer và random giá trị row, column
    # print(kernel)
        self.results = np.zeros((int((self.input.shape[0] - self.kernel.shape[1])/self.stride) + 1,
                                 int((self.input.shape[1] - self.kernel.shape[2])/self.stride) + 1,
                                 self.kernel.shape[0])) # do result đầu ra là ảnh nên số layer đặt sau cùng

    def getROI(self):
        for row in range(0, int((self.input.shape[0] - self.kernel.shape[1])/self.stride) + 1):
            for column in range(0, int((self.input.shape[1] - self.kernel.shape[2])/self.stride) + 1):
                roi = self.input[row*self.stride: row*self.stride + self.kernel.shape[1],
                      column*self.stride: column*self.stride +self.kernel.shape[2]]
                yield row, column, roi

    def operate(self):
        for layer in range(self.kernel.shape[0]):
            for row, column, roi in self.getROI():
                self.results[row,column,layer] = np.sum(roi * self.kernel[layer]) # lấy roi nhân với từng layer

        return self.results



class ReLU:
      def __init__(self,input):   #input là kết quả đã được convolution
          self.input = input
          self.result = np.zeros((self.input.shape[0],self.input.shape[1],
                                  self.input.shape[2]))

      def operate(self):
          for layer in range(self.input.shape[2]):
              for row in range(self.input.shape[0]):
                for column in range(self.input.shape[1]):
                  self.result[row,column,layer] = 0 if self.input[row, column,layer] < 0 else self.input[row,column,layer]

          return self.result


class MaxPooling:
    def __init__(self, input, poolingSize=2):
        self.input = input
        self.poolingSize = poolingSize
        self.result = np.zeros((int(self.input.shape[0]/self.poolingSize),
                                int(self.input.shape[1]/self.poolingSize),
                                self.input.shape[2])) # số layer input truyền vào
    def operate(self):
        for layer in range(self.input.shape[2]):
            for row in range(int(self.input.shape[0]/self.poolingSize)):
                for column in range(int(self.input.shape[1]/self.poolingSize)):
                    #Lấy max trong từng lần chập chạy qua trong mỗi layer
                    self.result[row, column, layer] =np.max( self.input[row*self.poolingSize: row*self.poolingSize + self.poolingSize,
                                                                 column*self.poolingSize: column*self.poolingSize + self.poolingSize,
                                                                 layer]) # Khi row=0,poolingSize =2,dòng  0*2 đi đén dòng 0*2 + 2

        return self.result

conv2d = convolution2d(img_gray,16,3,padding=0,stride=1)
fig = plt.figure(figsize=(10,10))
img_gray_conv2d = conv2d.operate()
img_gray_conv2d_relu = ReLU(img_gray_conv2d).operate()
# Từ RLu cho chạy qua max pooling
img_gray_conv2d_relu_MaxPooling = MaxPooling(img_gray_conv2d_relu,3).operate()
#Do số kernel là 16 nên kết quả sẽ là 16 tấm ảnh , nên dùng vòng lặp
for i in range(16):
    plt.subplot(4,4,i+1) # tạo ra khung hình 4 dòng 4 cột
    plt.imshow(img_gray_conv2d_relu_MaxPooling[:, :, i],cmap='gray')
    plt.axis('off')

    plt.savefig('img_gray_conv2d_Relu_MaxPooling.jpg')


plt.show()
#conv2d_relu = ReLU(img_gray_conv2d) #truyền input vào là kết quả đã qua convolution
#img_gray_conv2d_relu = conv2d_relu.operate()
# hiển thị ra hình ảnh
#plt.imshow(img_gray_conv2d_relu)
#plt.show()
