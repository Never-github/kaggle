from skimage.io import imread
from skimage.filters import sobel
from skimage.filters import gaussian
import numpy as np
def BlurOrNot(path):
    img = imread(path)
    #img_blur = gaussian(img)

    mag_ord = sobel(img)
    #mag_blur = sobel(img_blur)
    def Sum_matrix(matrix):
        sum=0
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                sum+=matrix[i][j]
        return sum/(len(matrix)*len(matrix[0]))
    avg_ori = Sum_matrix(mag_ord)
    avg_ori = np.sum(avg_ori)
    score = avg_ori
    #print(maxv)
    print(avg_ori)
    print(f'the image get the score: {score}')
    # mag_blur, angle_blur = cv2.cartToPolar(gx_blur, gy_blur, angleInDegrees=True)
    #avg_blur = (Sum_matrix(mag_blur))
    #avg_blur = np.sum(avg_blur)
    #print(avg_blur)
    if score>0.15:
        print('It is not blur!!')
        return False
    else:
        print('It is blur!!!')
        return True

BlurOrNot(path = 'images/4.jpg')