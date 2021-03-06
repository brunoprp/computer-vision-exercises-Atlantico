# Solutions for Optical flow and tracking
## 1\) Optical flow e tracking
Selecionar uma área de um video e realizar o tracking utilizando Optical Flow. Desenhe o vetor resultante entre as localizações das features.


<a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/optical_flow/output/opticalflow_dense.png"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/optical_flow/output/opticalflow_dense..png"></a>  
 
Foi implementado a resolução em dois ambientes Python: Spyder com uma versão do código `optical_flow.py` e uma versão no Jupyter Notebook  `optical_flow.ipynb`. Ambas as implementações tem as mesma funcionalidades, onde foi criado uma classe `OpticalFlow( )` que possui dois métodos: `lucasKenedMethod(self, video_path, tracking_point, n_frames)`  responsável por rastrear um ponto selecionado no vídeo, os parâmetros, corresponde ao caminho do arquivo em vídeo em seguida um array com as coordenadas `X` e `Y`  do ponto de rastreio do primeiro frame do vídeo e por último é definido a quantidade de frames que será usado no rastreio do ponto. Já o método `ImgChooseTracking(self, video_path)` pode ser usado para ajudar a encontrar um ponto de interesse no primeiro frame do vídeo, o único parâmetro é o caminho do arquivo em vídeo, Para teste e analisar o resultado execute o arquivo [optical_flow.ipynb](https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/1-Optical-Flow/optical_flow.ipynb)  ou o arquivo [optical_flow.py](https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/1-Optical-Flow/optical_flow.py).

### Python Version 3.7 
##### Installation of used libraries:
```sh
$ pip install numpy==1.16.2
$ pip install opencv-python==4.1.2
$ pip install matplotlib==3.1.3

```
#### Results:
 <a href="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/1-Optical-Flow/result-images/result1.png"><img src="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/1-Optical-Flow/result-images/result1.png"  width="550" alt="accessibility text"></a>  
 
  <a href="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/1-Optical-Flow/result-images/result2.png"><img src="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/1-Optical-Flow/result-images/result2.png"  width="550" alt="accessibility text"></a> 
