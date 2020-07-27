# Solutions for Image stitching
## 2\) Image stitching
  * Utilizar descritores de imagem, como SURF, SIFT ou ORB para identificar descritores similares entre imagens e conecta-las, gerando uma única imagem.


  <a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image1.jpg"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image1.jpg" align="left" width="128"></a>



  <a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image2.jpg"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image2.jpg" align="left" width="128"></a>



  <a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image3.jpg"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image3.jpg" align="left" width="128"></a>



  <a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image4.jpg"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image4.jpg" align="left" width="128"></a>

  

  <a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image5.jpg"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/image_stitching/assets/image5.jpg" width="128"></a>


Foi implementado a resolução em dois ambientes Python: Spyder com uma versão do código [image_stitching.py]() e uma versão no Jupyter Notebook [Image_stitching.ipynb](). Ambas as implementações tem as mesma funcionalidades, onde foi criado uma classe ImageStitching()  que possui o  métodos stitchingImg(self, img1,img2, method)  responsável por fazer a junção de duas imagens por vez. Na parte inferior do cadigo existe uma implantação Image stitching com uma função pronta do OpenCv stitchingOpencv(list_dire_images), capaz de fazer fazer a junção de várias imagens simultaneamente, mais informações [aqui](https://docs.opencv.org/master/d2/d8d/classcv_1_1Stitcher.html). Para teste e analisar o resultado execute o arquivo [Image_stitching.ipynb](https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/2-Image-Stitching/Image_stitching.ipynb) ou o [image_stitching.py](https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/2-Image-Stitching/image_stitching.py).

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
   
   



 


