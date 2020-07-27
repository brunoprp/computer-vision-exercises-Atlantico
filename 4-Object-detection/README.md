# Object detection with Pytorch
## 1\) Object detection com deep learning
Detectar objetos em imagens ou vídeo utilizando um método de deep learning.

<a href="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/object_detection/output/output.png"><img src="https://raw.githubusercontent.com/alanoMartins/computer_vision_exercises/master/object_detection/output/output.png"></a>  

### Pacotes necessários para a implementação e execução deste repositório
* Python 3.7.7
* Pytorch 1.3.1
* Torchvision 0.4.2
* OpenCv-Python 4.1.2.20
* OpenCv-Contrib-Python 4.3.0.36
* Pillow 7.2.0
* Jsonlib 0.9.5
* Itertools 8.4.0
* Numpy 1.18.5
* Matplotlib 3.1.3
#### Instalaçãoes
Pytorch on GPU Anaconda
```sh 
$ conda install -c anaconda pytorch-gpu
```
Pytorch on CPU Anaconda
```sh 
$ conda install -c anaconda pytorch-cpu
```
Torchvision on Anaconda
```sh 
$ conda install -c pytorch torchvision 
```
```sh
$ pip install numpy==1.16.2
$ pip install opencv-python==4.1.2
$ pip install matplotlib==3.1.3
$ pip install opencv-contrib-python
$ pip install Pillow
$ pip install jsonlib
$ pip install more-itertools
```

## Implementation
Essa implementação consiste em um modelo de detecção de objetos  treinado do zero, uma variantes do modelo [Single Short Multibox Detector (SSD) ](https://arxiv.org/abs/1512.02325), utilizando uma rede pré treinada para extração de características VGG16 (Base Convolutions).
Essa implementação é uma adaptação dos códigos de dois tutoriais que pode ser encontrados aqui, [Tutorial 1](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html), [Tutorial 2](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#training).


### 1\) Download dataset 
   O dataset usado foi uma adaptação do [CAMVID](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) da versão 2007 e 2012, o dataset usado  pode ser baixado em [aqui](https://drive.google.com/file/d/1J470_BPkD4lvcfBqQzoDWYSkOYyatM22/view?usp=sharing) 
### 2\) Data preparation
   Após baixado e descompactado o dataset, execute o arquivo `preparation_data.py` passando o caminho da pasta do dataset baixado, com isso será criando as listas de treino e teste `.json`, exemplo:
   ```python
   from utils import create_data_lists

   if __name__ == '__main__':
       create_data_lists(voc07_path='/media/bruno/HD-Arquivos2/Data_Object_Detect/VOC2007',
                         voc12_path='/media/bruno/HD-Arquivos2/Data_Object_Detect/VOC2012',
                         output_folder='/media/bruno/HD-Arquivos2/Data_Object_Detect/')
           
```
### 3\) Model training
   Para retreinar um modelo execute o arquivo `model_train.py` ou `model_train.ipynb` onde pode ser analisado o último treinamento realizado, no código passe o caminho da pasta onde foi criado as listas `.json`, exemplo:  
```python 
data_folder = '/media/bruno/HD-Arquivos2/Data_Object_Detect/'
```

  Os hiperparâmetros  usados são os mesmo sugerido pelo [Papel](https://arxiv.org/abs/1512.02325) com exceção do  `batch_size = 15` e `epocas = 20` o otmizador usado é [SGD](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD). Os hiperparâmetros são localizados na parte superior do código para facilitar a modificação caso precise, o formato de entrada na rede para treinamneto é de imagens `300x300` RGB. O treinamento foi realizado localmente usando uma placa de vídeo Nvidia Geforce 1060 de 6 GB, o tempo médio de treinamento foi de 6 minutos para cada época. O treinamento também pode ser realizado em plataformas de machine learning como [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true) e [Kaggle](https://www.kaggle.com/), basta fazer o Upload do dataset.   
  
  A metrica usado para avaliar o desempenho de treinado é a função de perda [MultiBox Loss](https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab) em `model.py` `MultiBoxLoss()` é uma métrica popular na medição da precisão de detectores de objetos como R-CNN mais rápido, SSD, etc.
### 4\) Evaluate Model
 Para iniciar a avaliação do modelo treinado é preciso executar o código `Evaluate_Model.py` ou `Evaluate_Model.ipynb` onde pode ser analizado a ultima avalição. Para isso na variável `checkpoint`, passe o caminho do arquivo salvo com o modelo treinado, exemplo   `checkpoint = 'checkpoint_ssd300.pth.tar`. As previsões analisadas são avaliadas em relação aos objetos de verdade fundamentais. A métrica de avaliação é a [Precisão Média Média (mAP)](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173), é usado a função `calculate_mAP()` em `utils.py` para esta finalidade. O modelo treinado por este repositório pode ser baixado [aqui](https://drive.google.com/file/d/1HBq4fsq7VyiZZqWllb3ZuDFZ9hIcNXUk/view?usp=sharing), para avaliar essa modelo execute o arquivo `Evaluate_Model.py` ou `Evaluate_Model.ipynb`, colocando o caminho da pasta do modelo baixado na variável  `checkpoint = 'checkpoint_ssd300.pth.tar`.
### 5\) Results 
* Abaixo está a curva de aprendizado com as 20 épocas de treinamento realizado, como pode-se perceber modelo melhora o seu semplendo gradativamente a cada época, com a avaliação da métrica mAP. O resultado da métrica mAP para os dados de testes foi 0.6427 para a média de todo as classe. Já para cada classe foi:

*  aeroplane = 0.708512008190155
*  bicycle = 0.7454192638397217
*  bird = 0.6230981945991516
*  boat = 0.5221483111381531
*  bottle = 0.24800018966197968
*  bus = 0.7201552391052246
*  car = 0.7631471753120422
*  cat = 0.8401749730110168
*  chair = 0.3495808243751526
*  cow = 0.703272819519043
*  diningtable = 0.6119518876075745
*  dog = 0.7849552035331726
*  horse = 0.8075953722000122
*  motorbike = 0.7407485842704773
*  person = 0.658084511756897
*  pottedplant = 0.2637472748756408,
*  sheep = 0.6770304441452026
*  sofa = 0.6690605878829956
*  train = 0.7741146087646484
*  tvmonitor = 0.6433150172233582


<a href="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/4-Object-detection/results/loss.png?raw=true"><img src="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/4-Object-detection/results/loss.png?raw=true"></a>

* Para analisar o resultado da detecção execute o arquivo example_image_detect.ipynb, passando o caminho do modelo na variável `checkpoint = 'checkpoint_ssd300.pth.tar'`  e o caminho da imagem  que será testada na variável   `img_path = 'image-test/img2.jpg'`, a baixo está alguns exemplos dessa execução:

  <a href="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/4-Object-detection/results/img-test1.png?raw=true"><img src="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/4-Object-detection/results/img-test1.png?raw=true"></a>
  
<a href="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/4-Object-detection/results/img-test2.png?raw=true"><img src="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/4-Object-detection/results/img-test2.png?raw=true"></a>

<a href="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/4-Object-detection/results/img-test3.png?raw=true"><img src="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/4-Object-detection/results/img-test3.png?raw=true"></a>

<a href="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/4-Object-detection/results/img-test4.png?raw=true"><img src="https://github.com/brunoprp/computer-vision-exercises-Atlantico/blob/master/4-Object-detection/results/img-test4.png?raw=true"></a>

### 6\) Limitations
   Para resultados mais concretos é preciso fazer mais testes com diferentes hiperparâmetros e aumentar o número de épocas de treinamento para um resultado melhor, porém isso não foi possível devido o tempo limitado para realização do trabalho. Também seria interessante testar  outros modelos pré-treinados para a extração de caracterís. 

