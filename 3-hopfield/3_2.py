import numpy as np
import itertools
from hopfield_net import HopfieldNet
import matplotlib.pyplot as plt


def main():

    images=load_data()
    threeimages = images[:3]
    net = HopfieldNet(max_iter=200)
    net.fit(threeimages)
    preds= net.predict(threeimages)

    accuracy = calc_element_accuracy(threeimages, preds)
    print('accuracy on training data: ', accuracy)

    degraded_image = images[9]
    image_pred= net.predict(degraded_image)

    origimage = np.reshape(threeimages[0], (1, -1))
    image_pred =np.reshape(image_pred, (1, -1))
    accuracy = calc_element_accuracy(origimage, image_pred)

    print('accuracy of degraded image p10 to p1: ', accuracy)

    degraded_image = images[10]
    image_pred= net.predict(degraded_image)

    origimage1 = np.reshape(threeimages[1], (1, -1))
    origimage2 = np.reshape(threeimages[2], (1, -1))
    image_pred =np.reshape(image_pred, (1, -1))
    accuracy1 = calc_element_accuracy(origimage1, image_pred)
    accuracy2 = calc_element_accuracy(origimage2, image_pred)

    accuracy3 = calc_element_accuracy(threeimages[1:3], image_pred)

    print('accuracy of degraded image p11 to p2 is: ', accuracy1) 
    print('accuracy of degraded image p11 to p3 is: ', accuracy2)
    print('accuracy of degraded image p11 to p3 and p2 is: ', accuracy3)


    #test sequential

    degraded_image = images[9]
    net2 = HopfieldNet(max_iter=2)
    net2.fit(threeimages)
    image_pred = net2.predict(degraded_image, method='sequential')

    origimage = np.reshape(threeimages[0], (1, -1))
    accuracy = calc_element_accuracy(origimage, image_pred)
    print('accuracy of sequential update is:', accuracy)

    snapshots = np.array(net2.hundreth_images)
    print(len(snapshots))
    
    for i in range(len(snapshots)):
        showimage(np.reshape(snapshots[i], (1,-1)))
        
    

    

    





def load_data():
    with open('pict.dat', 'r') as f:
        text = str(f.read())
        value_list = np.array([int(val) for val in text.split(',')])
        images=[]
        for n in range(11):
            start_index = 1024*n
            end_index=1024*(n+1)
            images.append(value_list[start_index:end_index])
        
        return np.array(images)


def showimage(image):
    image = np.reshape(image, (32, 32)).T
    plt.imshow(image)
    plt.show()

def calc_element_accuracy(patterns, preds):
    n_total = patterns.shape[0] * patterns.shape[1]
    n_correct = np.sum(patterns == preds)
    return n_correct / n_total

def calc_pattern_accuracy(patterns, preds):
    n_total = patterns.shape[0]
    n_correct = 0
    for pattern, pred in zip(patterns, preds):
        if (pattern == pred).all():
            n_correct += 1
    return n_correct / n_total




  





main()
   
