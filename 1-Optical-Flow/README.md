# Solutions for Optical flow e tracking
## 1\) Optical flow e tracking
Selecionar uma área de um video e realizar o tracking utilizando Optical Flow. Desenhe o vetor resultante entre as localizações das features.

Utilizar imagens da câmera do dispositivo (notebook ou celular) local, caso não tenha acesso, baixar um vídeo de exemplo e anexar no resultado.
- parse:

<a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/optical_flow/output/opticalflow_dense.png"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/optical_flow/output/opticalflow_dense..png"></a>

- Dense:

<a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/optical_flow/output/opticalflow_sparse.jpg"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/optical_flow/output/opticalflow_sparse.jpg" width=320></a>
  
 
###### Install dependences:

```sh
$ pip install opencv-contrib-python
$ pip install numpy
```
Usage libs
```python
import matplotlib.pyplot as plt
import numpy as np
import cv2
```
Function for calcl optical flow 
```python
cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
```
#### class Optical_flow()
methods
```python
#função para definir os parametros inicias 
def __init__(self):
    pass
#função para calcular o fluxo óptico com o método de Lucas-Kanade, via OpenCv 
def calc_lucas_kened(self,old_gray,frame,p0): #Recebe o frame anterior e o frame atual
    return img #retorna a imagem já calculada 
```