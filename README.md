# Low-Light Image Enhancement Network Using Informative Feature Stretch and Attention
# Abstract
Low-light images often exhibit reduced brightness, weak contrast, and color distortion. Consequently, enhancing low-light images is essential to make them suitable for computer vision tasks. Nevertheless, addressing this task is particularly challenging because of the inherent constraints posed by low-light environments. In this study, we propose a novel low-light image enhancement network using adaptive feature stretching and informative attention. The proposed network architecture mainly includes an adaptive feature stretch block designed to extend the narrow range of image features to a broader range. To achieve improved image restoration, an informative attention block is introduced to assign weight to the output features from the adaptive feature stretch block. We conduct comprehensive experiments on widely used benchmark datasets to assess the effectiveness of the proposed network. The experimental results show that the proposed low-light image enhancement network yields satisfactory results compared with existing state-of-the-art methods from both subjective and objective perspectives while maintaining acceptable network complexity.
# Requirement
    - python 3.9.15
    - pytorch 1.10.1
    - You can find other details in requirements.txt
# Testing
    1. Create a test_img folder and put the data you want to test in it.
    2. run test.py
# Citation
if you find this project useful, please cite our paper:
https://doi.org/10.3390/electronics13193883

    @article{FSANet2024,
      title={Low-Light Image Enhancement Network Using Informative Feature Stretch and Attention},
      author={Sung Min Chun, Jun Young Park, Il Kyu Eom},
      journal={Electronics},
      year={2024}      
    }
# Contact
If you have any questions, please contact smin5874@pusan.ac.kr.
