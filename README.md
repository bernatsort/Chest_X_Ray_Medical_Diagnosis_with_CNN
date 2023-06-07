# Diagnóstico de COVID-19 y neumonía a partir de radiografías de tórax

## Objetivos: 
### 1. Clasificación de partes del cuerpo
La primera etapa del proyecto consiste en clasificar los 22 diferentes tipos de partes del cuerpo utilizando el modelo VGG16. Las 22 partes del cuerpo, con sus correspondientes etiquetas, son:
  Abdomen
• Tobillo
• Vértebras Cervicales
• Pecho
• Clavículas
• Codo
• Pies
• Dedo
• Antebrazo
• Mano
• Cadera
• Rodilla
• Espinilla
• Vértebras Lumbares
• Otros
• Pelvis
• Hombro
• Nariz
• Cráneo
• Pierna
• Vértebras Torácicas
• Muñeca
### 2. Clasificación de imágenes de pulmón
La segunda parte de este proyecto consiste en realizar un modelo que, a partir de imágenes de rayos X del pulmón, nos saque un diagnóstico sobre si el paciente tiene COVID19, neumonía, o por el contrario se trata de un paciente sano. 

## Datasets: 
1. The UNIFESP X-Ray Body Part Classification Dataset
- https://www.kaggle.com/datasets/felipekitamura/unifesp-xray-bodypart-classification

2. Chest X-ray (Covid-19 & Pneumonia)
- original: https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia?resource=download-directory
- with the validation set: https://drive.google.com/drive/folders/1rXe2zRIbl_dr0T0FvDMfsoo2HlmlX87B?usp=sharing

## Notebooks y scripts: 
### Clasificación de partes del cuerpo
1. Body_Classification.py
### Clasificación de imágenes de pulmón
1. create_validation_set.py
2. EDA_lung_classif.ipynb
3. covid19_pneumonia_classification.ipynb
