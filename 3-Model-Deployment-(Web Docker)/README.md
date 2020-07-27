# Solutions for Model Deployment (Web/Docker)
## 3\) Model Deployment (Web/Docker)
  * Implementar serviço web básico com uma rota que receba imagem via file upload e retorne o resultado processado.

  <a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/web-service/input.jpg"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/web-service/input.jpg"  width="400"></a>  <a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/web-service/postman-output.png"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/web-service/postman-output.png"  width="530"></a>
  
 Para essa implementação foi usado a [ DeepLabV3](https://towardsdatascience.com/review-deeplabv3-atrous-convolution-semantic-segmentation-6d818bfd1d74)  da Google com a bibliotecas [TensorFlow 2](https://www.tensorflow.org/) e [Keras 2](https://keras.io/), O DeepLabV3 é um modelo de aprendizado profundo usado para segmentação semântica de imagens. Esse modelo é usado para fazer a segmentação da imagem. A aplicação Web foi feita usando o framework web para Python [Flask](https://flask.palletsprojects.com/en/1.1.x/). O arquivo [web_app.py](https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/3-Model-Deployment-(Web%20Docker)/web_app.py) é responvavel por fazer a segmentação e remoção do  background da imagem com o modelo DeepLabV3 veja a função `imagePreprocessing(pathImage)` que recebe a imagem via Web. Também no arquivo  [web_app.py](https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/3-Model-Deployment-(Web%20Docker)/web_app.py) existe as funções: `rootPage()` responsável de criar a página Web para fazer o Upload da imagem, já a função `uploadFile()` é responsável de receber a imagem e usar a função  `imagePreprocessing(pathImage)`   para realizar o processamento na imagem e retorna uma página Web com a imagem processada. Também foi criado um container usando Docker para execução da aplicação em um ambiente separado. 

 
### Python Version 3.7 
##### Installation of used libraries:
```sh
$ pip install numpy==1.16.2
$ pip install opencv-python==4.1.2
$ pip install matplotlib==3.1.3
$ pip install opencv-contrib-python

```
#### Results:

   <a href="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/2-Image-Stitching/imges_results/panorama.jpg?"><img src="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/2-Image-Stitching/imges_results/panorama.jpg?" align="left" width="600"></a>
   
 
 
   
   <a href="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/2-Image-Stitching/imges_results/all_panorama.jpg?"><img src="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/2-Image-Stitching/imges_results/all_panorama.jpg?" align="left" width="600"></a>
   
   



 


