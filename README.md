AI+ Cell Segmentation Project


Cells are the fundamental building blocks of all living organisms. Over the past few decades, advancements in microscopy techniques have revolutionized our ability to study cellular structures in detail. One critical step in this process is cell segmentation, which is essential for analyzing cell features, dynamics, migration, and morphology. Two of the most advanced and widely used software tools for this purpose are CellPose and StarDist, both of which leverage deep learning techniques to enhance segmentation accuracy and efficiency. They are capable of cell segmentation for nucleus image, cell membrane image and label-free images. 
              

In this project, we would like to explore the applicability of these two software and study the performance over publicly available datasets.

Mentor: Liya Ding 


A stretch goal of the project is to train data-specific model for data from Gardel Lab. 2D or 3D models trained from pre-trained model and fine tuned with given dataset and see if 

Software:
Cellpose documentation and code available at https://www.cellpose.org/ and https://github.com/MouseLand/cellpose .
StarDist documentation and code available at https://stardist.net/ and https://github.com/stardist/stardist/ .

Public available dataset with annotation:
LIVECell dataset (label-free images)
      https://sartorius-research.github.io/LIVECell
DeepCell dataset (different types of tissue cells)
         https://datasets.deepcell.org/data
Evaluation metrics:
Intersection over Union (IoU)
Precision and Accuracy
(Reference: A systematic evaluation of computation methods for cell segmentation,Y. Wang et al. doi: https://doi.org/10.1101/2024.01.28.577670 )
