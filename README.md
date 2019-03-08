# Hazing and Dehazing image processing project

## Dehazing - Color Attenuation

Implementation of Fast Single Image Haze Removal Algorithm Using Color Attenuation Prior.

### [Research Paper](https://github.com/jacobjohn2016/Hazing-and-Dehazing-Project/blob/master/qingsongzhu2015.pdf)
Q. Zhu, J. Mai, and L. Shao ,"A Fast Single Image Haze Removal Algorithm Using Color Attenuation Prior", in IEEE Transactions On Image Processing, Vol. 24, NO. 11, NOVEMBER 2015, pp. 3522-3533

### Dependencies
1. Python 3.6+
2. Python Packages
   * numpy
   * cv2
   * matplotlib

### Instructions
    folder structure:
        dehazed.jpg # dehazed output
        vit.jpg # sample input image
        room.jpeg
        dehaze.py # main
    
    To dehaze:
        run python3 dehaze.py vit.jpg the output 'dehazed.jpg' will be saved in the same folder

### Contributions
* _Original author_ - [TummanapallyAnuraag](https://github.com/TummanapallyAnuraag)
* _Made code reproducible_ - [jacobjohn2016](https://github.com/jacobjohn2016)

## Dehazing - Dark Channel Prior
This program implement single image dehazing using dark channel prior. 

## Research Papers
[He, Kaiming, Jian Sun, and Xiaoou Tang. "Single image haze removal using dark channel prior." IEEE transactions on pattern analysis and machine intelligence 33.12 (2011): 2341-2353.](https://github.com/jacobjohn2016/Hazing-and-Dehazing-Project/blob/master/kaiminghe2011.pdf)
[He, Kaiming, Jian Sun, and Xiaoou Tang. "X.: Guided image filtering." In: ECCV. 2010.](https://github.com/jacobjohn2016/Hazing-and-Dehazing-Project/blob/master/he2010.pdf)

### Dependencies
1. Python 3.7
2. Python Packages
   * numpy
   * cv2
   * matplotlib

### Instructions
    folder structure:
        image # image folder
            14.png
            15.png # main image
            16.png
            J.png
        dehaze.py # file to execute
    
    To dehaze:
        1. save your image file as 15.png in the ./image folder
        2. run python3 dehaze.py the output 'J.png' will be saved in the same folder

## Examples
<center>
<img src="./image/15.png"  height = "400" alt="Before Image" />
<img src="./image/J.png"   height = "400" alt="After Dehazing" />
</center>

## [Blog](http://www.freethatphoto.com/how-dehazing-works-photo/)
- How dehazing works: a simple explanation

### Contributions
* _Original author_ - [He-Zhang](https://github.com/He-Zhang/image_dehaze)
* _Made code reproducible_ - [jacobjohn2016](https://github.com/jacobjohn2016)

## Hazing

### [Research Paper](https://github.com/jacobjohn2016/Hazing-and-Dehazing-Project/blob/master/zhang2017.pdf)
Zhang, Ning, Lin Zhang, and Zaixi Cheng. "Towards Simulating Foggy and Hazy Images and Evaluating Their Authenticity." International Conference on Neural Information Processing. Springer, Cham, 2017.

### License:
    This code is made publicly for research use only. 
    It may be modified and redistributed under the terms of the GNU General Public License.
    Please cite the paper and source code if you are using it in your work.

### Dependencies
1. Python 3.6+
2. Python Packages
   * numpy
   * cv2
   * math
   * matplotlib
   * tool_kit
   * scipy.io 
   * noise
   * PIL

### Instructions:  
    folder structure:
          FoHIS/const.py  # define const
                fog.py  # main
                parameter.py # all parameters used in simulating fog/haze are defined here.
                tool_kit.py # some useful functions
                
          AuthESI/compute_aggd.py
                  compute_authenticity.py  # main
                  guided_filter.py  # some functions
                  prisparam_16_hazeandfog.mat  # pre-trained model
                  
          img/img.jpg  # RGB image
              imgd.jpg  # depth image
              result.jpg  # simulation
              
    1. To simulate fog/haze effects:
        run python FoHIS/fog.py, the output 'result.jpg' will be saved in ../img/
          
    2. To evaluate the authenticity:
        run python compute_authenticity.py to evaluate 'result.jpg' in ../img/
                  

### Dataset:
![image](https://github.com/jacobjohn2016/Hazing-and-Dehazing-Project/blob/master/Hazing/img/dataset.png)

| Source Image  | Maximum Depth | Effect | Homogeneous | Particular Elevation|
|:-------------:|:-------------:|:------:|:-----------:|:-------------------:|
| (a)           |     150 m     | Haze   |      Yes    |         No          |
| (b)           |     400 m     | Haze   |      Yes    |         No          |
| (c)           |     800 m     | Haze   |      Yes    |         No          |
| (d)           |     30 m      | Fog    |      Yes    |         No          |
| (e)           |     150 m     | Fog    |      No     |         Yes         |
| (f)           |     30 m      |Fog+Haze|      No     |         No          |
| (g)           |     600 m     | Haze   |      Yes    |         No          |
| (h)           |     400 m     | Haze   |      Yes    |         No          |
| (i)           |     200 m     | Haze   |      Yes    |         No          |
| (j)           |     100 m     | Haze   |      Yes    |         No          |
| (k)           |     100 m     | Haze   |      Yes    |         No          |
| (l)           |     800 m     |Fog+Haze|      No     |         Yes         |
| (m)           |     300 m     | Haze   |      Yes    |         No          |
| (n)           |     60 m      | Haze   |      Yes    |         No          |
| (o)           |     300 m     | Haze   |      Yes    |         No          |
| (p)           |     1000 m    | Haze   |      Yes    |         No          |
| (q)           |     400 m     | Haze   |      Yes    |         No          |
| (r)           |     300 m     | Haze   |      Yes    |         No          |

### Contributions
* _Original author_ - [noahzn (Noah)](https://github.com/noahzn)
* _Made code reproducible_ - [jacobjohn2016](https://github.com/jacobjohn2016)
