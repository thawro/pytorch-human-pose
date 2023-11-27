
## Intro
* https://learnopencv.com/human-pose-estimation-using-keypoint-rcnn-in-pytorch/

## In short
### Top-down
### Bottom-up



## Classic (old) approaches
| Link | Year | Title | Approach | Features |
| -------: | :------: | :------: | :------: | :------ |
| [link](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) | 2004 | Distinctive Image Features from Scale-Invariant Keypoints | SIFT | TODO |
| [link](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) | 2005 | Histograms of Oriented Gradients for Human Detection | HoG | TODO |
| [link](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/BodyPartRecognition.pdf) | 2011 | Real-Time Human Pose Recognition in Parts from Single Depth Images| TODO | TODO |
| [link](https://www.ri.cmu.edu/pub_files/2014/7/poseMachines.pdf) | 2013 | Pose Machines: Articulated Pose Estimation via Inference Machines| TODO | TODO |

## Human Pose
| Link | Year | Title | Approach | Features |
| -------: | :------: | :------: | :------: | :-------------- |
| [link](https://arxiv.org/pdf/1312.4659.pdf) | 2013 | DeepPose: Human Pose Estimation via Deep Neural Networks| AlexNet + Linear | AlexNet as backbone, Linear as last (predicting n_keypoints * 2), Cascade of regressors (next regressors minimize residua), L2 loss |
| [link](https://arxiv.org/pdf/1603.06937.pdf) | 2016 | Stacked Hourglass Networks for Human Pose Estimation| Stacked Hourglass Networks | A single hourglass consists of the bottom-up (high-res to low-res) conv layers and top-down (low-res to high-res via upsampling) conv layers (done with residual connections). Each hourglass produces a heatmap for each keypoint and the loss (heatmaps MSE) is calculated for each hourglass. The hourglasses are meant to refine the predictions |
| [link](https://arxiv.org/pdf/1602.00134.pdf) | 2016 | Convolutional Pose Machines| Pose Machines | Pose belief maps (heatmaps) are predicted at different stages of CNN network and passed to the next stages (first stage has own CNN params, next stages have shared CNN params). Loss is calculated for each belief map. Differs from hourglass by not introducing enc-dec architecture |
| [link](https://arxiv.org/pdf/1605.03170v3.pdf) | 2016 | DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model| TODO | TODO |
| [link](https://arxiv.org/pdf/1611.08050v2.pdf) | 2017 | **Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields**| Part Affinity Fields (PAF) | Multistage neural network which predicts keypoints and Part Affinity Fields (PAFs) heatmaps after each stage. The next stages take as input not only image features but also those heatmaps. PAFs are representing the vectors pointing from one keypoint to another. Complex post-processing is done to ensure good multi-person pose prediction  |
| [link](https://arxiv.org/pdf/1611.05424.pdf) | 2017 | Associative Embedding: End-to-End Learning for Joint Detection and Grouping | Associative Embedding | Bottom-up multiperson human pose estimation with novel keypoint grouping method based on tags (vector embeddings). Stacked Hourglass network is used to estimate the detection heatmaps and associative embedding heatmaps for each keypoint (2K heatmaps). MSE loss is calculated for detection heatmaps. The loss for embeddings aims to provide similar embedding for same person instance and different embeddings for different person instances. The final grouping is done using embeddings with similar values (small distance). Also instance segmentation task is conducted.|
| [link](https://arxiv.org/pdf/1701.01779.pdf) | 2017 | Towards Accurate Multi-person Pose Estimation in the Wild | G-RMI | TODO |
| [link](https://arxiv.org/pdf/1711.08229.pdf) | 2017 | Integral Human Pose Regression | TODO | TODO |
| [link](https://arxiv.org/pdf/1711.07319.pdf) | 2018 | **Cascaded Pyramid Network for Multi-Person Pose Estimation** | Cascaded Pyramid Network (CPN)  | CPN includes a GlobalNet based on the feature pyramid structure and a RefineNet which concatenates all the pyramid features as a context information. In addition, online hard mining is integrated in RefineNet to explicitly address the “hard” keypoints. L2 Loss is calculated for both Global and RefineNet |
| [link](https://arxiv.org/pdf/1804.06208.pdf) | 2018 | Simple Baselines for Human Pose Estimation and Tracking| TODO | Simple network with Deconv layers on top of last ResNet features. Three deconv layers are added to improve the spatial resolution. The network predicts single human pose, but works on top of Region Proposal Network (RPN) which predicts the human bboxes. Authors also introduced an algorithm for pose tracking in the videos. Tracking is done with pose shift via optical flow field, bbox estimation and greedy person ID matching (using specified similarity metric) |
| [link](https://arxiv.org/pdf/1807.04067.pdf) | 2018 | MultiPoseNet: Fast Multi-Person Pose Estimation using Pose Residual Network | MultiPoseNet | Bottom-up multi-person pose estimation NN, that combines a multi-task (person detection, keypoint detection, person segmentation and pose estimation problems) model with a novel keypoints assignment method. Keypoint assignment is implemented by the Pose Residual Network (residual MLP) which receives keypoints and person detections, and produces accurate poses by assigning keypoints to person instances |
| [link](https://arxiv.org/pdf/1612.00137.pdf) | 2018 | RMPE: Regional Multi-Person Pose Estimation | RMPE | Top-down regional multi-person pose estimation (RMPE) framework. RMPE facilitates pose estimation the presence of inaccurate human bounding boxes. It consists of three components: Symmetric Spatial Transformer Network (SSTN), Parametric Pose NMS, and Pose-Guided Proposals Generator (PGPG). SSTN extracts high quality dominant human proposals. Parametric NMS defines NMS parameters via data-centric approach. PGPG augments the dataset by adding offsets to human bboxes. The offsets are determined by data distribution (via k-means). |
| [link](https://arxiv.org/pdf/1803.08225.pdf) | 2018 | PersonLab: Person Pose Estimation and Instance Segmentation with a Bottom-Up, Part-Based, Geometric Embedding Model | PersonLab | Bottom-up model that employs a conv net which learns to detect individual keypoints and predict their relative displacements, allowing to group keypoints into person pose instances. Further, a part-induced geometric embedding descriptor is proposed which allows to associate semantic person pixels with their corresponding person instance, delivering instance-level person segmentations. Different type of heatmaps are estimated: keypoints heatmap, segmentation and short-range, mid-range, and long-range offsets. The short/mid-range offsets are used to refine the predicted keypoints and group into instances. The long-range offests are used to perform instance segmentation | 
| [link](https://arxiv.org/pdf/1805.09707v1.pdf) | 2018 | Jointly Optimize Data Augmentation and Network Training: Adversarial Data Augmentation in Human Pose Estimation | TODO | TODO |
| [link](https://arxiv.org/pdf/1902.09212v1.pdf) | 2019 | **Deep High-Resolution Representation Learning for Human Pose Estimation**| **HRNet** | HRNet maintains high-resolution representations through the whole process. Starting from a high-res subnet as the 1st stage, gradually add high-to-low-res subnets one by one to form more stages. Multi-res subnets are connected in parallel and multi-scale functions are conducted such that each of the high-to-low res representations receives information from other parallel representations over and over leading to rich high-res representations. As a result, the predicted keypoint heatmap is potentially more accurate and spatially more precise |
| [link](https://arxiv.org/pdf/1904.07850v2.pdf) | 2019 | **Objects as Points** | **CenterNet** | Models an object as a single point — the center point of its bounding box. Our detector uses keypoint estimation to find center points and regresses to all other object properties, such as size, 3D location, orientation, and pose. Model predicts objects center points [C], model size [2] (in real pixel values) and center point offsets [2] and uses all those predictions to estimate final bounding box |
| [link](https://arxiv.org/pdf/1812.08008.pdf) | 2019 | **OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields**| Part Affinity Fields (PAF) | Multistage neural network which predicts keypoints and Part Affinity Fields (PAFs) heatmaps sequentially. The first stages predict and refine PAF maps. Next stages takes as input image feature and last refined PAF map and predict Parts confidence heatmaps. Person poses are composed using greedy algorithm which connects the parts using values calculated with PAF maps (complex stuff) |
| [link](https://arxiv.org/pdf/1812.00324.pdf) | 2019 | CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark | CrowdPose | Combined top-down and bottom-up approach. First the human bboxes are detected. Then for each bbox the SPPE network is run to estimate the keypoints. The SPPE is learned in a way to predict on only the persons keypoints but also keypoints belonging to other persons in the bbox. When all keypoints (from all bboxes) are predicted, the Person-Joint graph is created and the global association algorithm is run to group the keypoints to each person. Also the new CrowdPose dataset is provided to ensure that there is a benchmark with crowded scenes. |
| [link](https://arxiv.org/pdf/1908.09220v1.pdf) | 2019 | **Single-Stage Multi-Person Pose Machines** | **Single-stage multi-person Pose Machine (SPM)** | A Single-stage multi-person pose estimation method (Structured Pose Representation, SPR), that unifies person instance and body joints positions representations. SPR introduces the root joints to indicate different person instances and joints positions (keypoints) are encoded into their displacements w.r.t. the roots. The method is further improved by introducing Hierarchical SPR, which encoded joints as displacements w.r.t. to joints from lower hierarchy level. Also the 3D human pose estimation is done on CMU Panoptic dataset. |
| [link](https://arxiv.org/pdf/1910.06278v1.pdf) | 2019 | Distribution-Aware Coordinate Representation for Human Pose Estimation | DARK | Proposed a more principled distribution-aware decoding method and improved the standard coordinate encoding process (i.e. transforming ground-truth coordinates to heatmaps) by generating accurate heatmap distributions for unbiased model training. Standard approach introduced quantization error to encoding and decoding. This approach gets rid of this error. Serving as a model-agnostic plugin, DARK significantly improves the performance of a variety of state-of-the-art human pose estimation model |
| [link](https://arxiv.org/pdf/1902.07837v3.pdf) | 2019 | Cascade Feature Aggregation for Human Pose Estimation | TODO | TODO |
| [link](https://arxiv.org/pdf/1908.10357.pdf) | 2020 | HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation| HigherHRNet | Bottom-up human pose estimation method for learning scale-aware representations using high-resolution feature pyramids. It uses multi-resolution supervision for training and multiresolution keypoints heatmaps aggregation for inference. The feature pyramid in HigherHRNet consists of feature map outputs from HRNet and upsampled higher-resolution outputs through a transposed convolution. Keypoints grouping is done by a method named associate embedding, which assigns each keypoint with a “tag” (a vector representation) and groups keypoints based on the L2 distance between tag vectors |
| [link](https://arxiv.org/pdf/1908.07919v2.pdf) | 2020 | Deep High-Resolution Representation Learning for Visual Recognition| HRNet | Great, in detail explanation of HRNet architecture |
| [link](https://arxiv.org/pdf/2006.10204.pdf) | 2020 | BlazePose: On-device Real-time Body Pose tracking| TODO | TODO |
| [link](https://arxiv.org/pdf/2003.04030v3.pdf) | 2020 | Learning Delicate Local Representations for Multi-Person Pose Estimation | TODO | TODO |
| [link](https://arxiv.org/pdf/2002.11098v1.pdf) | 2020 | Toward fast and accurate human pose estimation via soft-gated skip connections | TODO | TODO |
| [link](https://arxiv.org/pdf/2001.08095v1.pdf) | 2020 | UniPose: Unified Human Pose Estimation in Single Images and Videos | UniPose | TODO |
| [link](https://arxiv.org/pdf/2104.02300.pdf) | 2021 | Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression| TODO | TODO |
| [link](https://arxiv.org/pdf/2107.00782v2.pdf) | 2021 | Polarized Self-Attention: Towards High-quality Pixel-wise Regression| TODO | TODO |
| [link](https://arxiv.org/pdf/2002.00537v2.pdf) | 2021 | Towards High Performance Human Keypoint Detection | TODO | TODO |
| [link](https://arxiv.org/pdf/2103.10180v1.pdf) | 2021 | OmniPose: A Multi-Scale Framework for Multi-Person Pose Estimation | OmniPose | TODO |
| [link](https://arxiv.org/pdf/2204.06806.pdf) | 2022 | YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss| TODO | TODO |
| [link](https://arxiv.org/pdf/2211.03375.pdf) | 2022 | AlphaPose: Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time| TODO | TODO |
| [link](https://arxiv.org/pdf/2204.12484v3.pdf) | 2022 | ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation | ViTPose | TODO |
| [link](https://arxiv.org/pdf/2303.11638v1.pdf) | 2023 | Human Pose as Compositional Tokens | TODO | TODO |
| [link](https://arxiv.org/pdf/2306.07879v2.pdf) | 2023 | Rethinking pose estimation in crowds: overcoming the detection information bottleneck and ambiguity | TODO | TODO |
| [link](TODO) | TODO | TODO | TODO | TODO |
| [link](TODO) | TODO | TODO | TODO | TODO |
| [link](TODO) | TODO | TODO | TODO | TODO |
| [link](TODO) | TODO | TODO | TODO | TODO |




## 3D Human Pose
| Link | Year | Title | Approach | Features |
| -------: | :------: | :------: | :------: | :-------------- |
| [link](https://arxiv.org/pdf/1905.05754v1.pdf) | 2019 | Learnable Triangulation of Human Pose | TODO | TODO |
| [link](https://arxiv.org/pdf/2004.06239v4.pdf) | 2020 | VoxelPose: Towards Multi-Camera 3D Human Pose Estimation in Wild Environment | VoxelPose | TODO |
| [link](https://dineshreddy91.github.io/papers/TesseTrack.pdf) | 2020 | TesseTrack: End-to-End Learnable Multi-Person Articulated 3D Pose Tracking | TesseTrack | TODO |
| [link](https://arxiv.org/pdf/2010.13302v1.pdf) | 2020 | AdaFuse: Adaptive Multiview Fusion for Accurate Human Pose Estimation in the Wild | AdaFuse | TODO |
| [link](https://arxiv.org/pdf/2109.05885v1.pdf) | 2021 | Graph-Based 3D Multi-Person Pose Estimation Using Multi-View Images | TODO | TODO |
| [link](https://arxiv.org/pdf/2111.04076v2.pdf) | 2021 | Direct Multi-view Multi-person 3D Pose Estimation | TODO | TODO |
| [link](https://arxiv.org/pdf/2104.02273v1.pdf) | 2021 | Multi-View Multi-Person 3D Pose Estimation with Plane Sweep Stereo | TODO | TODO |
| [link](https://arxiv.org/pdf/2207.10955v1.pdf) | 2022 | Faster VoxelPose: Real-time 3D Human Pose Estimation by Orthographic Projection | Faster VoxelPose | TODO |
| [link](https://dl.acm.org/doi/abs/10.1145/3528233.3530746) | 2022 | QuickPose: Real-time Multi-view Multi-person Pose Estimation in Crowded Scenes | QuickPose | TODO |
| [link](TODO) | TODO | TODO | TODO | TODO |



## Face landmarks
| Link | Year | Title | Approach | Features |
| -------: | :------: | :------: | :------: | :-------------- |
| [link](TODO) | TODO | TODO | TODO | TODO |
| [link](TODO) | TODO | TODO | TODO | TODO |
| [link](TODO) | TODO | TODO | TODO | TODO |
| [link](TODO) | TODO | TODO | TODO | TODO |
| [link](TODO) | TODO | TODO | TODO | TODO |
| [link](https://arxiv.org/pdf/2106.04269v2.pdf) | 2021 | HPRNet: Hierarchical Point Regression for Whole-Body Human Pose Estimation | HPRNet | TODO |
| [link](https://arxiv.org/pdf/2007.03221v3.pdf) | 2021 | AnchorFace: An Anchor-based Facial Landmark Detector Across Large Poses | AnchorFace | TODO |
| [link](https://arxiv.org/pdf/2306.02763v1.pdf) | 2023 | STAR Loss: Reducing Semantic Ambiguity in Facial Landmark Detection | TODO | TODO |


## Hand Pose
| Link | Year | Title | Approach | Features |
| -------: | :------: | :------: | :------: | :-------------- |
| [link](https://arxiv.org/pdf/1704.07809.pdf) | 2017 | Hand Keypoint Detection in Single Images using Multiview Bootstrapping| TODO | TODO |
| [link](https://arxiv.org/pdf/2007.09590v1.pdf) | 2020 | AWR: Adaptive Weighting Regression for 3D Hand Pose Estimation | AWR | TODO |
| [link](https://arxiv.org/pdf/2108.05545v1.pdf) | 2021 | HandFoldingNet: A 3D Hand Pose Estimation Network Using Multiscale-Feature Guided Folding of a 2D Hand Skeleton | HandFoldingNet | TODO |
| [link](https://ieeexplore.ieee.org/document/9319235) | 2021 | Hand Pose Estimation in the Task of Egocentric Actions | TODO | TODO |
| [link](https://arxiv.org/pdf/2203.15458v1.pdf) | 2022 | Efficient Virtual View Selection for 3D Hand Pose Estimation | Virtual View | TODO |
| [link](https://arxiv.org/pdf/2206.07117v2.pdf) | 2022 | TrihornNet: A model for accurate depth-based 3D hand pose estimation | TrihornNet | TODO |
| [link](https://arxiv.org/pdf/1908.09999v1.pdf) | 2019 | A2J: Anchor-to-Joint Regression Network for 3D Articulated Pose Estimation from a Single Depth Image | A2J | TODO |
| [link](TODO) | TODO | TODO | TODO | TODO |
| [link](TODO) | TODO | TODO | TODO | TODO |
| [link](TODO) | TODO | TODO | TODO | TODO |
| [link](TODO) | TODO | TODO | TODO | TODO |
| [link](TODO) | TODO | TODO | TODO | TODO |




# Datasets

| Name | Year | Train | Val | Test | Labels  | Info | Task | 
| -------: | :------: | :------: | :------: | :------: | :------: | :------: | :------ |
| [CMU](http://domedb.perception.cs.cmu.edu/index.html) | 2017 | TODO |  TODO | TODO | TODO | TODO  | Human pose |
| [MPII](http://human-pose.mpi-inf.mpg.de/) | 2014 | TODO |  TODO | TODO | TODO | [link](http://human-pose.mpi-inf.mpg.de/contents/andriluka14cvpr.pdf)  | Human Pose |
| [FLIC](https://bensapp.github.io/flic-dataset.html) | 2013 | TODO |  TODO | TODO | TODO | TODO  | Human Pose |
| [Helen](http://www.ifp.illinois.edu/~vuongle2/helen/) | 2012 | 2000 |  330 | TODO | TODO | [link]()  | Face Landmark |
| [300-W](https://ibug.doc.ic.ac.uk/resources/300-W/) | 2013 | TODO |  TODO | TODO | TODO | [link]()  | Face Landmark |
| [300-VW](https://ibug.doc.ic.ac.uk/resources/300-VW/) | 2015 | TODO |  TODO | TODO | TODO | [link]()  | Face Landmark |
| [NYU Hand Pose](https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm) | 2014 | TODO |  TODO | TODO | TODO | [link]()  | Hand Pose |
| [MSRA Hand](https://paperswithcode.com/dataset/msra-hand) | 2014 | TODO |  TODO | TODO | TODO | [link]()  | Hand Pose |
| [LS Multiview 3D](http://www.rovit.ua.es/dataset/mhpdataset/#download) | 2017 | TODO |  TODO | TODO | TODO | [link](https://arxiv.org/pdf/1707.03742.pdf)  | Hand Pose |
| [MS COCO](https://cocodataset.org/#download) | 2017 | TODO |  TODO | TODO | TODO | [link]()  | Human Pose |
| [HANDS 2019](https://docs.google.com/forms/d/e/1FAIpQLSdp6sZoml5aMaiZixqG9wfGOS9Hvz6Lh1aFRYPs18HC2BDuSA/viewform) | 2019 | TODO |  TODO | TODO | TODO | [link]()  | Hand Pose |
| [TODO]() | TODO | TODO |  TODO | TODO | TODO | [link]()  | Keypoints detection |
| [TODO]() | TODO | TODO |  TODO | TODO | TODO | [link]()  | Keypoints detection |
| [TODO]() | TODO | TODO |  TODO | TODO | TODO | [link]()  | Keypoints detection |


