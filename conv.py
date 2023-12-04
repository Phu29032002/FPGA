import cv2
import matplotlib.pyplot as plt
import numpy as np

laplacian = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])


class Conv2d:
    def __init__(self, input,kernel, kernelsize):
        self.input = input
        self.kernel = kernel
        self.height, self.width = input.shape
        #self.kernel = np.array([[0, 1, 0],
         #                       [1, -4, 1],
          #                      [0, 1, 0]])
        self.result = np.zeros((self.height - kernelsize + 1, self.width - kernelsize + 1))

    def getRoi(self, input):
        for row in range(self.height - self.kernel.shape[0] + 1):
            for col in range(self.width - self.kernel.shape[1] + 1):
                roi = input[row:row + self.kernel.shape[0], col:col + self.kernel.shape[1]]
                yield roi, row, col

    def operate(self):
        for roi, row, col in self.getRoi(self.input):
            self.result[row, col] = np.sum(roi * self.kernel)
        return self.result


# relu
class Relu:
    def __init__(self, input):
        self.input = input
        self.height, self.width = input.shape
        self.result = np.zeros((self.height, self.width))

    def operate(self):
        for row in range(self.input.shape[0]):
            for col in range(self.input.shape[1]):
                if self.input[row, col] > 0:
                    self.result[row, col] = self.input[row, col]
                else:
                    self.result[row, col] = 0
        return self.result

class MaxPooling:
    def __init__(self, input, kernelsize):
        self.input = input
        self.kernelsize = kernelsize
        self.height, self.width = input.shape
        self.result = np.zeros((self.height - kernelsize + 1, self.width - kernelsize + 1))

    def getRoi(self, input):
        for row in range(self.height - self.kernelsize + 1):
            for col in range(self.width - self.kernelsize + 1):
                roi = input[row:row + self.kernelsize, col:col + self.kernelsize]
                yield roi, row, col

    def operate(self):
        for roi, row, col in self.getRoi(self.input):
            self.result[row, col] = np.max(roi)
        return self.result



#pattern
img_pattern = cv2.imread("pattern.jpg")
#img = cv2.resize(img, (200, 200))
img_gray_pattern = cv2.cvtColor(img_pattern, cv2.COLOR_BGR2GRAY)
print(img_gray_pattern.shape)

conv2d_pattern = Conv2d(img_gray_pattern, laplacian,3 )
img_gray_conv2d_pattern = conv2d_pattern.operate()
conv2d_relu_pattern = Relu(img_gray_conv2d_pattern)
conv2d_relu_img_pattern = conv2d_relu_pattern.operate()

print(conv2d_relu_img_pattern.shape)

#input
img_input = cv2.imread("imgaeinput.jpg")
img_input = cv2.resize(img_input, (200, 200))
img_gray_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
print(img_gray_input.shape)

conv2d_input = Conv2d(img_gray_input, laplacian,3)
img_gray_conv2d_input = conv2d_input.operate()
conv2d_relu_input = Relu(img_gray_conv2d_input)
conv2d_relu_img_input = conv2d_relu_input.operate()
print(conv2d_relu_img_input.shape)

#convol pattern and input
conv2d_stage2 = Conv2d(conv2d_relu_img_input,  conv2d_relu_img_pattern,26)
img_gray_conv2d_stage2 = conv2d_stage2.operate()
conv2d_relu_stage2 = Relu(img_gray_conv2d_stage2)
conv2d_relu_img_stage2 = conv2d_relu_stage2.operate()
print(conv2d_relu_img_stage2.shape)

#max pooling conv2d_relu_img_stage2
maxpooling_stage2 = MaxPooling(conv2d_relu_img_stage2, 2)
maxpooling_img_gray_stage2 = maxpooling_stage2.operate()

# Plotting side by side
plt.subplot(1, 2, 1)
plt.imshow(img_gray_pattern, cmap='gray')
plt.title('Original Pattern')

plt.subplot(1, 2, 2)
plt.imshow(conv2d_relu_img_pattern, cmap='gray')
plt.title('Processed Pattern')

plt.show()

plt.subplot(1, 2, 1)
plt.imshow(img_gray_input, cmap='gray')
plt.title('Original Input')

plt.subplot(1, 2, 2)
plt.imshow(conv2d_relu_img_input, cmap='gray')
plt.title('Processed Input')

plt.show()

plt.imshow(conv2d_relu_img_stage2, cmap='gray')
plt.title('stage 2')

plt.show()


plt.imshow(maxpooling_img_gray_stage2, cmap='gray')
plt.title('Max pooling')

plt.show()
#plt.imshow(conv2d_relu_img_pattern, cmap='gray')
#plt.show()
