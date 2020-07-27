# Solutions for Image stitching
## 2\) Image stitching
  * Utilizar descritores de imagem, como SURF, SIFT ou ORB para identificar descritores similares entre imagens e conecta-las, gerando uma única imagem.


  <a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image1.jpg"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image1.jpg" align="left" width="128"></a>



  <a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image2.jpg"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image2.jpg" align="left" width="128"></a>



  <a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image3.jpg"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image3.jpg" align="left" width="128"></a>



  <a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image4.jpg"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image4.jpg" align="left" width="128"></a>

  

  <a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image5.jpg"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image5.jpg" width="128"></a>


Foi implementado a resolução em dois ambientes Python: Spyder com uma versão do código `optical_flow.py` e uma versão no Jupyter Notebook  `optical_flow.ipynb`. Ambas as implementações tem as mesma funcionalidades, onde foi criado uma classe `OpticalFlow( )` que possui dois métodos: `lucasKenedMethod(self, video_path, tracking_point, n_frames)`  responsável por rastrear um ponto selecionado no vídeo, os parâmetros, corresponde ao caminho do arquivo em vídeo em seguida um array com as coordenadas `X` e `Y`  do ponto de rastreio do primeiro frame do vídeo e por último é definido a quantidade de frames que será usado no rastreio do ponto. Já o método `ImgChooseTracking(self, video_path)` pode ser usado para ajudar a encontrar um ponto de interesse no primeiro frame do vídeo, o único parâmetro é o caminho do arquivo em vídeo, Para teste e analisar o resultado execute o arquivo [optical_flow.ipynb](https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/1-Optical-Flow/optical_flow.ipynb)  ou o arquivo [optical_flow.py](https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/1-Optical-Flow/optical_flow.py).

### Python Version 3.7 
##### Installation of used libraries:
```sh
$ pip install numpy==1.16.2
$ pip install opencv-python==4.1.2
$ pip install matplotlib==3.1.3

```
#### Results:
 * Input image:
 
<a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image1.jpg"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image1.jpg" align="left" width="200"></a>
     
<a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image2.jpg"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image2.jpg" align="left" width="200"></a>
    
* Output image:
 
<a href="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/2-Image-Stitching/imges_results/panorama.jpg?raw=true"><img src="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/2-Image-Stitching/imges_results/panorama.jpg?raw=true" align="left" width="600"></a>

 
 
 


