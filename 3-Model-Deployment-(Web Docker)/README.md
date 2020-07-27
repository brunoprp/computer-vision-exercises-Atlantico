# Solutions for Model Deployment (Web/Docker)
## 3\) Model Deployment (Web/Docker)
  * Implementar serviço web básico com uma rota que receba imagem via file upload e retorne o resultado processado.

  <a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/web-service/input.jpg"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/web-service/input.jpg"  width="400"></a>  <a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/web-service/postman-output.png"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/web-service/postman-output.png"  width="530"></a>
  
 Para essa implementação foi usado a [ DeepLabV3](https://towardsdatascience.com/review-deeplabv3-atrous-convolution-semantic-segmentation-6d818bfd1d74)  da Google com a bibliotecas [TensorFlow 2](https://www.tensorflow.org/) e [Keras 2](https://keras.io/), O DeepLabV3 é um modelo de aprendizado profundo usado para segmentação semântica de imagens. Esse modelo é usado para fazer a segmentação da imagem. A aplicação Web foi feita usando o framework web para Python [Flask](https://flask.palletsprojects.com/en/1.1.x/). O arquivo [web_app.py](https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/3-Model-Deployment-(Web%20Docker)/web_app.py) é responvavel por fazer a segmentação e remoção do  background da imagem com o modelo DeepLabV3 veja a função `imagePreprocessing(pathImage)` que recebe a imagem via Web. Também no arquivo  [web_app.py](https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/3-Model-Deployment-(Web%20Docker)/web_app.py) existe as funções: `rootPage()` responsável de criar a página Web para fazer o Upload da imagem, já a função `uploadFile()` é responsável de receber a imagem e usar a função  `imagePreprocessing(pathImage)`   para realizar o processamento na imagem e retorna uma página Web com a imagem processada. Também foi criado um container usando Docker para execução da aplicação em um ambiente separado. 

 
### Execution instructions
1\) Faça o clone do repositorio 
```sh
$ git clone https://github.com/brunoprp/computer-vision-exercises-Atlantico.git
```
2\) Navegue até a pasta da aplicação
```sh
$ cd computer-vision-exercises-Atlantico/3-Model-Deployment-(Web Docker)
```
3\) Execute o arquivo Dockerfile, supondo que o Docker já esteja instalado, caso não [aqui](https://www.digitalocean.com/community/tutorials/como-instalar-e-usar-o-docker-no-ubuntu-18-04-pt) um tutorial de como instalar.
```sh
$ docker build -t web_app:latest .
```
3\) Faça execução do container para exultar a aplicação.
```sh
$ docker run -i -p 5000:5000 web_app:latest
```
Com isso as instalações dos pacotes necessários serão realizadas, após isso é só testar a aplicação via POSTMAN ou pelo  navegador, abaixa está alguns exemplos de como testar a aplicação. 


#### Results:

   <a href="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/3-Model-Deployment-(Web%20Docker)/imgs-results/resul1.png?raw=true"><img src="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/3-Model-Deployment-(Web%20Docker)/imgs-results/resul1.png?raw=true" align="left" width="600"></a>
   
 
 
   
   <a href="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/3-Model-Deployment-(Web%20Docker)/imgs-results/resul2.png"><img src="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/3-Model-Deployment-(Web%20Docker)/imgs-results/resul2.png" align="left" width="600"></a>
   
   <a href="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/3-Model-Deployment-(Web%20Docker)/imgs-results/result4.png?raw=true"><img src="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/3-Model-Deployment-(Web%20Docker)/imgs-results/result4.png?raw=true" align="left" width="600"></a>
   
   <a href="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/3-Model-Deployment-(Web%20Docker)/imgs-results/result5.png?raw=true"><img src="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/3-Model-Deployment-(Web%20Docker)/imgs-results/result5.png?raw=true" align="left" width="600"></a>
   
   



 


