# face_masks_detection_opencv_tf

![image.png](https://github.com/vivekpatel99/face_masks_detection_opencv_tf/blob/master/results/result1.png)
**Context**
This dataset is used for Face Mask Detection Classification with images. The dataset consists of almost 12K images which
are almost 328.92MB in size.

**Acknowledgments**
All the images with the face mask (~6K) are scrapped from google search and all the images without the face mask are
preprocessed from the CelebFace dataset created by Jessica Li (https://www.kaggle.com/jessicali9530). Thank you so much
Jessica for providing a wonderful dataset to the community.

**Inspiration**
The inspiration behind creating this dataset is to create an algorithm that can directly detect is a person is wearing a
face mask or not. So I've scrapped the images from google as well as from the CelebFace dataset created by Jessica
Li (https://www.kaggle.com/jessicali9530) to make this happen.

The dataset is donwloaded from
kaggle [Face Mask Detection ~12K Images Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)

## Steps to solve the problem

1. Develop and train CNN model to detect face
2. Label faces
    1. Using Haarcascade find and crop faces
    2. using CNN model to predict the result
    3. label the faces with predicted result


## HowTo use the script
1. Create a virtual environment for python either using anaconda or venv 
2. install requirements.txt file using `$$ pip install -r requirements.txt` command
3. run the jupyter notebooks
   
![image.png](https://github.com/vivekpatel99/face_masks_detection_opencv_tf/blob/master/results/result2.png)
