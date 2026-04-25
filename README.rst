============================
Energy-based Flow Classifier
============================

The Energy-Based Flow Classifier (EFC) is a new classification method developed in the context of network intrusion detection systems. It was first presented in
`A New Method for Flow-Based Network Intrusion Detection Using the Inverse Potts Model <https://ieeexplore.ieee.org/document/9415676>`_ and latter developed in `A novel open set energy-based flow classifier for network intrusion detection <https://www.sciencedirect.com/science/article/pii/S0167404825002585>`_. 

Dependencies
------------

EFC package requires:

- Python (>= 3.8)
- Cython (>= 0.29)
- NumPy (>= 1.21.4)
- Scikit-learn (>= 1.0.1)
- joblib (>= 1.1.0)
- threadpoolctl (>= .0.0)

Installation
------------

Currently, the only way to install EFC is from source, using its GitHub repository. To do so, clone this repository and run the following commands::

    git clone https://github.com/EnergyBasedFlowClassifier/EFC-package
    cd EFC-package
    pip install -r requirements.txt
    pip install .


Usage
-----
Use EFC like a scikit-learn estimator::

    from efc import EnergyBasedFlowClassifier

    clf = EnergyBasedFlowClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

It supports both binary and multiclass classification.
When the target is binary, EFC is a single-class algorithm. Therefore, the user must choose which class will be used as the base class.
When the target is multiclass, the user must choose whether to use the unknown label or not. 


The EFC internally normalizes and discretizes the input data. However, like any scikit-learn estimator, it requires categorical features to be encoded before input. It is also necessary that categorical columns are specified when calling the fit method, so that they are ignored during attribute preprocessing.
For a full explanation of each of EFC's parameters, read the `API documentation <https://efc-package.readthedocs.io/en/latest/generated/efc.EnergyBasedFlowClassifier.html#efc.EnergyBasedFlowClassifier>`_ in Read the Docs.

Citations
---------

If you use EFC in a scientific publication, please cite the respective papers::

    @article {9415676,
    author={Pontes, Camila F. T. and de Souza, Manuela M. C. and Gondim, João J. C. and Bishop, Matt and Marotta, Marcelo Antonio},
    journal={IEEE Transactions on Network and Service Management},
    title={A New Method for Flow-Based Network Intrusion Detection Using the Inverse Potts Model},
    year={2021},
    volume={18},
    number={2},
    pages={1125-1136},
    doi={10.1109/TNSM.2021.3075503}}

    @article{SOUZA2025104569,
    author = {Manuela M.C. Souza and Camila T. Pontes and João J.C. Gondim and Luís P.F. Garcia and Luiz DaSilva and Eduardo F.M. Cavalcante and Marcelo A. Marotta},
    journal = {Computers & Security},
    title = {A novel open set energy-based flow classifier for network intrusion detection},
    year = {2025},
    issn = {0167-4048},
    pages = {104569},
    doi = {https://doi.org/10.1016/j.cose.2025.104569},
    url = {https://www.sciencedirect.com/science/article/pii/S0167404825002585}}

Related Works
-------------
    [1] "C. F. T. Pontes, M. M. C. de Souza, J. J. C. Gondim, M. Bishop and M. A. Marotta, *A New Method for Flow-Based Network Intrusion Detection Using the Inverse Potts Model*, in *IEEE Transactions on Network and Service Management*, vol. 18, no. 2, pp. 1125-1136, June 2021, doi: 10.1109/TNSM.2021.3075503."

    [2] "J. M. De Almeida et al., *Abnormal Behavior Detection Based on Traffic Pattern Categorization in Mobile Cellular Networks*, in *IEEE Transactions on Network and Service Management*, doi: 10.1109/TNSM.2021.3125019."

    [3] "Manuela M.C. Souza and Camila T. Pontes and João J.C. Gondim and Luís P.F. Garcia and Luiz DaSilva and Eduardo F.M. Cavalcante and Marcelo A. Marotta, *A novel open set energy-based flow classifier for network intrusion detection*, in *Computers & Security*, issn 0167-4048, pp. 104569, June 2025, doi: 10.1016/j.cose.2025.104569."

Other authors that cited this work [according to IEEE]:

    [4] "L. H. d. Melo, G. d. C. Bertoli, M. Nogueira, A. L. d. Santos and L. A. Pereira, *Anomaly-Flow: A Multi-Domain Federated Generative Adversarial Network for Distributed Denial-of-Service Detection*, in *IEEE Network*, vol. 40, no. 2, pp. 269-277, March 2026, doi: 10.1109/MNET.2025.3567251."
    
    [5] "E. Rodrigues de Oliveira, R. de Oliveira Albuquerque and J. J. C. Gondim, *Methodology for Building Realistic Network Intrusion Detection Datasets: A Case Study with the Energy-Based Flow Classifier*, in *2025 Workshop on Communication Networks and Power Systems (WCNPS)*, Brasilia, Brazil, 2025, pp. 1-6, doi: 10.1109/WCNPS69127.2025.11295886."
    
    [6] "S. Kumar Nandi, R. Ratti, S. Ranbir Singh and S. Nandi, *Prompt Engineering-Based Network Intrusion Detection System*, in *IEEE Access*, vol. 13, pp. 190859-190871, 2025, doi: 10.1109/ACCESS.2025.3629761."
    
    [7] "H. Barakat, A. H. El Fawal, A. Mansour, A. Nasser and H. El Ghor, *Hierarchical DDoS Attacks Classification for IoT Networks Using Ensemble Learning*, in *2025 IEEE International Conference on Emerging Trends in Engineering and Computing (ETECOM)*, Riffa, Bahrain, 2025, pp. 1-6, doi: 10.1109/ETECOM66111.2025.11318988."
    
    [8] "Q. Xu, H. He, H. Yu, Z. Peng and L. Nie, *P4NFC: A P4-Based Comprehensive Network Flow Classification Scheme in SDN*, in *2025 IEEE International Conference on High Performance Computing and Communications (HPCC)*, Exeter, United Kingdom, 2025, pp. 632-639, doi: 10.1109/HPCC67675.2025.00098."
    
    [9] "S. M. Shivam, A. P V and P. Jayachandran, *Anomaly Detection in Cloud Networks using Machine Learning Techniques*, in *2025 3rd International Conference on Sustainable Computing and Data Communication Systems (ICSCDS)*, Erode, India, 2025, pp. 1054-1058, doi: 10.1109/ICSCDS65426.2025.11167399."
    
    [10] "M. Wan et al., *A Federated Learning-Based Intrusion Detection System for Satellite-Terrestrial Integrated Networks*, in *ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, Hyderabad, India, 2025, pp. 1-5, doi: 10.1109/ICASSP49660.2025.10890330."
    
    [11] "F. Duan, W. Cui, Y. Ma and J. Hu, *Network Intrusion Detection System Based on BERT-MLP Model*, in *2025 4th International Symposium on Computer Applications and Information Technology (ISCAIT)*, Xi'an, China, 2025, pp. 1861-1866, doi: 10.1109/ISCAIT64916.2025.11010372."
    
    [12] "S. Zhao, Y. Liu, Y. Zhang, X. Liu, J. Liu and Z. Liu, *Unsupervised Lightweight Intrusion Detection Using Federated Learning for Heterogeneous Network Environments*, in *2025 5th International Conference on Sensors and Information Technology*, Nanjing, China, 2025, pp. 628-634, doi: 10.1109/ICSI64877.2025.11009273."
    
    [13] "H. Liu, *ScCEE-IDS: A Federated Learning-Based Intrusion Detection System for Smart Cities with Cloud-Edge-End Integration*, in *2024 International Conference on Ubiquitous Computing and Communications (IUCC)*, Chengdu, China, 2024, pp. 542-547, doi: 10.1109/IUCC65928.2024.00099."
    
    [14] "P. Turaka and S. K. Panigrahy, *Dynamic Attack Detection in IoT Networks: An Ensemble Learning Approach With Q-Learning and Explainable AI*, in *IEEE Access*, vol. 12, pp. 161925-161940, 2024, doi: 10.1109/ACCESS.2024.3485989."
    
    [15] "A. A. Nassar and W. G. Morsi, *A Fast and Effective Automated Wavelet-Deep learning-based Method to Detect Cyberattacks in Microgrids with EV Fast Charging Stations*, in *2024 IEEE Canadian Conference on Electrical and Computer Engineering (CCECE)*, Kingston, ON, Canada, 2024, pp. 583-589, doi: 10.1109/CCECE59415.2024.10667144."
    
    [16] "S. Singh, A. K. Pandey, A. Kamal, S. Swain, M. K. Gourisaria and A. Bandyopadhyay, *E-commerce Trading using VCG Auction Mechanism in Metaverse*, in *2024 2nd World Conference on Communication & Computing (WCONF)*, RAIPUR, India, 2024, pp. 1-5, doi: 10.1109/WCONF61366.2024.10692297."
    
    [17] "Y. Zhu, C. Li and Y. Wang, *Network Intrusion Detection Scheme Based on Federated Learning in Heterogeneous Network Environments*, in *2024 13th International Conference on Communications, Circuits and Systems (ICCCAS)*, Xiamen, China, 2024, pp. 491-496, doi: 10.1109/ICCCAS62034.2024.10652649."
    
    [18] "Y. Liang, X. Lv, J. Wang and H. Zhao, *A Network Security Event Detection for Power Monitoring System Based on Matrix Factorization*, in *2024 9th International Conference on Intelligent Computing and Signal Processing (ICSP)*, Xian, China, 2024, pp. 251-257, doi: 10.1109/ICSP62122.2024.10743961."
    
    [19] "X. Wang, X. Wang, M. He, M. Zhang and Z. Lu, *Spatial-Temporal Graph Model Based on Attention Mechanism for Anomalous IoT Intrusion Detection*, in *IEEE Transactions on Industrial Informatics*, vol. 20, no. 3, pp. 3497-3509, March 2024, doi: 10.1109/TII.2023.3308784."
    
    [20] "Y. A. Rani, K. Deepthi Reddy and R. U. Rani, *A Novel Network Intrusion Detection Model using Residual Recurrent Neural Network with Improved Garter Snake-based Optimization Strategy*, in *2023 Global Conference on Information Technologies and Communications (GCITC)*, Bangalore, India, 2023, pp. 1-8, doi: 10.1109/GCITC60406.2023.10426136."
    
    [21] "N. Kaur, J. Singla, G. Mathur, S. Talwani and N. Malik, *An Advanced Feature Selection Approach to Improve Intrusion Detection System using Machine Learning*, in *2023 7th International Conference on Electronics, Communication and Aerospace Technology (ICECA)*, Coimbatore, India, 2023, pp. 984-992, doi: 10.1109/ICECA58529.2023.10394718."
    
    [22] "S. Pansare, A. Malik and I. Batra, *Comparative Analysis of Machine Learning based Intrusion Detection Systems*, in *2023 Seventh International Conference on Image Information Processing (ICIIP)*, Solan, India, 2023, pp. 474-479, doi: 10.1109/ICIIP61524.2023.10537635."
    
    [23] "A. Aouatif, B. Omar, C. Hakima and E. M. Abdelmajid, *An optimized neural network-based IDS against DDoS attacks*, in *2023 10th International Conference on Wireless Networks and Mobile Communications (WINCOM)*, Istanbul, Turkiye, 2023, pp. 1-6, doi: 10.1109/WINCOM59760.2023.10322910."
    
    [24] "N. Alsabilah and D. B. Rawat, *An Adaptive Flow-based NIDS for Smart Home Networks Against Malware Behavior Using XGBoost combined with Rough Set Theory*, in *2023 10th International Conference on Internet of Things: Systems, Management and Security (IOTSMS)*, San Antonio, TX, USA, 2023, pp. 15-22, doi: 10.1109/IOTSMS59855.2023.10325714."
    
    [25] "Y. Liu, A. Zhao, L. Du, C. Zhang, H. Yan and Z. Gu, *OIDMD: A Novel Open-Set Intrusion Detection Method Based on Mahalanobis Distance*, in *2023 8th International Conference on Data Science in Cyberspace (DSC)*, Hefei, China, 2023, pp. 253-260, doi: 10.1109/DSC59305.2023.00044."
    
    [26] "T. Ahmad and D. Truscan, *Efficient Early Anomaly Detection of Network Security Attacks Using Deep Learning*, in *2023 IEEE International Conference on Cyber Security and Resilience (CSR)*, Venice, Italy, 2023, pp. 154-159, doi: 10.1109/CSR57506.2023.10224923."
    
    [27] "G. Apruzzese, P. Laskov and J. Schneider, *SoK: Pragmatic Assessment of Machine Learning for Network Intrusion Detection*, in *2023 IEEE 8th European Symposium on Security and Privacy (EuroS&P)*, Delft, Netherlands, 2023, pp. 592-614, doi: 10.1109/EuroSP57164.2023.00042."
    
    [28] "L. G. Nguyen and K. Watabe, *A Method for Network Intrusion Detection Using Flow Sequence and BERT Framework*, in *ICC 2023 - IEEE International Conference on Communications*, Rome, Italy, 2023, pp. 3006-3011, doi: 10.1109/ICC45041.2023.10279335."
    
    [29] "M. Verkerken et al., *A Novel Multi-Stage Approach for Hierarchical Intrusion Detection*, in *IEEE Transactions on Network and Service Management*, vol. 20, no. 3, pp. 3915-3929, Sept. 2023, doi: 10.1109/TNSM.2023.3259474."
    
    [30] "A. Kumar and S. Kumar, *Intrusion detection based on machine learning and statistical feature ranking techniques*, in *2023 13th International Conference on Cloud Computing, Data Science & Engineering (Confluence)*, Noida, India, 2023, pp. 606-611, doi: 10.1109/Confluence56041.2023.10048802."
    
    [31] "X. Deng, J. Zhu, X. Pei, L. Zhang, Z. Ling and K. Xue, *Flow Topology-Based Graph Convolutional Network for Intrusion Detection in Label-Limited IoT Networks*, in *IEEE Transactions on Network and Service Management*, vol. 20, no. 1, pp. 684-696, March 2023, doi: 10.1109/TNSM.2022.3213807."
    
    [32] "O. Barut, Y. Luo, P. Li and T. Zhang, *R1DIT: Privacy-Preserving Malware Traffic Classification With Attention-Based Neural Networks*, in *IEEE Transactions on Network and Service Management*, vol. 20, no. 2, pp. 2071-2085, June 2023, doi: 10.1109/TNSM.2022.3211254."
    
    [33] "Y. -C. Lai et al., *Task Assignment and Capacity Allocation for ML-Based Intrusion Detection as a Service in a Multi-Tier Architecture*, in *IEEE Transactions on Network and Service Management*, vol. 20, no. 1, pp. 672-683, March 2023, doi: 10.1109/TNSM.2022.3203427."
    
    [34] "L. H. de Melo, G. de C Bertoli, L. A. Pereira, O. Saotome, M. F. Domingues and A. L. dos Santos, *Generalizing Flow Classification for Distributed Denial-of-Service over Different Networks*, in *GLOBECOM 2022 - 2022 IEEE Global Communications Conference*, Rio de Janeiro, Brazil, 2022, pp. 879-884, doi: 10.1109/GLOBECOM48099.2022.10001530."
    
    [35] "S. Neupane et al., *Explainable Intrusion Detection Systems (X-IDS): A Survey of Current Methods, Challenges, and Opportunities*, in *IEEE Access*, vol. 10, pp. 112392-112415, 2022, doi: 10.1109/ACCESS.2022.3216617."
    
    [36] "J. Bi, Z. Guan and H. Yuan, *Hybrid Network Intrusion Detection with Stacked Sparse Contractive Autoencoders and Attention-based Bidirectional LSTM*, in *2022 IEEE International Conference on Systems, Man, and Cybernetics (SMC)*, Prague, Czech Republic, 2022, pp. 6-11, doi: 10.1109/SMC53654.2022.9945600."
    
    [37] "D. A. G. Lopes, M. A. Marotta, M. Ladeira and J. J. C. Gondim, *Botnet detection based on network flow analysis using inverse statistics*, in *2022 17th Iberian Conference on Information Systems and Technologies (CISTI)*, Madrid, Spain, 2022, pp. 1-6, doi: 10.23919/CISTI54924.2022.9820318."
    
    [38] "I. A. Abdulmajeed and I. M. Husien, *Machine Learning Algorithms and Datasets for Modern IDS Design*, in *2022 IEEE International Conference on Cybernetics and Computational Intelligence (CyberneticsCom)*, Malang, Indonesia, 2022, pp. 335-340, doi: 10.1109/CyberneticsCom55287.2022.9865255."
    
    [39] "G. Apruzzese, L. Pajola and M. Conti, *The Cross-Evaluation of Machine Learning-Based Network Intrusion Detection Systems*, in *IEEE Transactions on Network and Service Management*, vol. 19, no. 4, pp. 5152-5169, Dec. 2022, doi: 10.1109/TNSM.2022.3157344."
    
    [40] "J. M. DeAlmeida et al., *Abnormal Behavior Detection Based on Traffic Pattern Categorization in Mobile Networks*, in *IEEE Transactions on Network and Service Management*, vol. 18, no. 4, pp. 4213-4224, Dec. 2021, doi: 10.1109/TNSM.2021.3125019."
    
    [41] "Y. Wei, J. Jang-Jaccard, F. Sabrina, A. Singh, W. Xu and S. Camtepe, *AE-MLP: A Hybrid Deep Learning Approach for DDoS Detection and Classification*, in *IEEE Access*, vol. 9, pp. 146810-146821, 2021, doi: 10.1109/ACCESS.2021.3123791."
    
    [42] "R. Badonnel, C. Fung, S. Scott-Hayward, Q. Li, J. Zhang and C. Hesselman, *Guest Editors’ Introduction: Special Issue on Latest Developments for Security Management of Networks and Services*, in *IEEE Transactions on Network and Service Management*, vol. 18, no. 2, pp. 1120-1124, June 2021, doi: 10.1109/TNSM.2021.3079189."
    
    [43] "M.S Harish, S Lokesh, P Sakthivel, B Akshaya, *Hybrid deep learning model for network intrusion detection using optimal feature fusion*, in *Ain Shams Engineering Journal*, vol. 17, no. 1, pp. 103904, 2026."
    
    [44] "Ashfaq Ahmad Najar et al., *LIDS: A Novel Lightweight Intrusion Detection System for DDoS Attacks*, in *SECURITY AND PRIVACY*, vol. 9, no. 1, 2026."
    
    [45] "Hui-Juan Zhang et al., *Reliable evaluation for the AI-enabled intrusion detection system from data perspective*, in *PLOS One*, vol. 20, no. 10, pp. e0334157, 2025."
    
    [46] "B. Selvakumar, B. Lakshmanan, J. R. Simeon, P. Ajith Kumar, *Feature selection using feature fusion based weighted multi objective Grey Wolf Optimization for network intrusion detection system*, in *Cluster Computing*, vol. 28, no. 14, 2025."
    
    [47] "Yuping Lai et al., *An Efficient Network Intrusion Detection Model Based on Beta Mixture Models*, in *Knowledge-Based Systems*, pp. 114506, 2025."
    
    [48] "Alice Bizzarri et al., *Neurosymbolic AI for network intrusion detection systems: A survey*, in *Journal of Information Security and Applications*, vol. 94, pp. 104205, 2025."
    
    [49] "Edison Prabhu. K, Godlin Debby K J, *Prediction of ddos attack by hierarchical attention-based bigru classifier*, in *Information Security Journal: A Global Perspective*, pp. 1, 2025."
    
    [50] "Prabu Kaliyaperumal et al., *Enhancing cybersecurity in Agriculture 4.0: A high-performance hybrid deep learning-based framework for DDoS attack detection*, in *Computers and Electrical Engineering*, vol. 126, pp. 110431, 2025."
    
    [51] "Manuela M.C. Souza et al., *A novel open set energy-based flow classifier for network intrusion detection*, in *Computers & Security*, pp. 104569, 2025."
    
    [52] "Hongpo Zhang et al., *Wasserstein distance guided feature Tokenizer transformer domain adaptation for network intrusion detection*, in *Computers & Security*, pp. 104562, 2025."
    
    [53] "Nikhil Sharma, Prashant Giridhar Shambharkar, *Multi-attention DeepCRNN: an efficient and explainable intrusion detection framework for Internet of Medical Things environments*, in *Knowledge and Information Systems*, 2025."
    
    [54] "Nouman Mabood et al., *A Comprehensive Survey on Software Defined Networking (SDN) Security*, in *Computing and Emerging Technologies*, vol. 2055, pp. 193, 2025."
    
    [55] "Jose Carlos Mondragon et al., *Advanced IDS: a comparative study of datasets and machine learning algorithms for network flow-based intrusion detection systems*, in *Applied Intelligence*, vol. 55, no. 7, 2025."
    
    [56] "Shitharth Selvarajan et al., *Diagnostic behavior analysis of profuse data intrusions in cyber physical systems using adversarial learning techniques*, in *Scientific Reports*, vol. 15, no. 1, 2025."
    
    [57] "Honglue Zhang et al., *Security Access Mechanism for Power Monitoring Network Based on PKI Public Key System*, in *The Proceedings of 2024 International Conference of Electrical, Electronic and Networked Energy Systems*, vol. 1330, pp. 441, 2025."
    
    [58] "Xinhang Li, Mingshu He, *Malicious Traffic Classification Algorithm Based on Multimodal Fusion*, in *Proceedings of the 3rd International Conference on Machine Learning, Cloud Computing and Intelligent Mining (MLCCIM2024)*, vol. 1327, pp. 229, 2025."
    
    [59] "Anil Kumar Gankotiya et al., *Cross-layer DDoS attack detection in wireless mesh networks using deep learning algorithm*, in *Journal of Electrical Engineering*, vol. 76, no. 1, pp. 34, 2025."
    
    [60] "Anthony Jacklingo Kwame Quansah Junior et al., *CDBi‐LSTM: A Hybrid Deep Learning Model With Attention‐Based Fusion for Efficient DDoS Detection in IoT Environments*, in *IET Communications*, vol. 19, no. 1, 2025."
    
    [61] "Meryem Janati Idrissi et al., *Flow timeout matters: Investigating the impact of active and idle timeouts on the performance of machine learning models in detecting security threats*, in *Future Generation Computer Systems*, pp. 107641, 2024."
    
    [62] "Li Yang, Chenglin Wen, *Multilevel identity fine authentication method based on time-frequency domain feature extraction in industrial internet of things system*, in *Journal of Control and Decision*, pp. 1, 2024."
    
    [63] "Mohammed Chemmakha et al., *Towards a Deep Learning Approach for IoT Attack Detection Based on a New Generative Adversarial Network Architecture and Gated Recurrent Unit*, in *Journal of Network and Systems Management*, vol. 32, no. 4, 2024."
    
    [64] "Abhay Pratap Singh et al., *Encrypted malware detection methodology without decryption using deep learning-based approaches*, in *Turkish Journal of Engineering*, vol. 8, no. 3, pp. 498, 2024."
    
    [65] "Ashfaq Ahmad Najar et al., *A novel CNN‐based approach for detection and classification of DDoS attacks*, in *Concurrency and Computation: Practice and Experience*, 2024."
    
    [66] "Liam Daly Manocchio et al., *A configurable anonymisation approach for network flow data: Balancing utility and privacy*, in *Computers and Electrical Engineering*, vol. 118, pp. 109465, 2024."
    
    [67] "Aklil Kiflay et al., *Network intrusion detection leveraging multimodal features*, in *Array*, pp. 100349, 2024."
    
    [68] "Ashfaq Ahmad Najar, Manohar Naik S., *A Robust DDoS Intrusion Detection System Using Convolutional Neural Network*, in *Computers and Electrical Engineering*, vol. 117, pp. 109277, 2024."
    
    [69] "Ali Shamekhi et al., *An intelligent behavioral-based DDOS attack detection method using adaptive time intervals*, in *Peer-to-Peer Networking and Applications*, 2024."
    
    [70] "R. Latha, R. M. Bommi, *Probability Boosted Regression for Intrusion Detection in Cyberactive Space*, in *Proceedings of the International Conference on Machine Learning, Deep Learning and Computational Intelligence for Wireless Communication*, pp. 247, 2024."
    
    [71] "Xiao Liao et al., *Load balancing and topology dynamic adjustment strategy for power information system network: a deep reinforcement learning-based approach*, in *Frontiers in Energy Research*, vol. 11, 2024."
    
    [72] "Sven Benjamin Kožić et al., *Rewiring driven evolution of quenched frustrated signed network*, in *Journal of Physics: Complexity*, vol. 5, no. 1, pp. 015001, 2024."
    
    [73] "Jing Bi et al., *Improved network intrusion classification with attention-assisted bidirectional LSTM and optimized sparse contractive autoencoders*, in *Expert Systems with Applications*, pp. 122966, 2023."
    
    [74] "Qingtian Zou et al., *Analysis of neural network detectors for network attacks*, in *Journal of Computer Security*, pp. 1, 2023."
    
    [75] "Gustavo Isaza et al., *DDoS Attacks Detection with Deep Learning Model Using a Cloud Architecture*, in *Trends in Sustainable Smart Cities and Territories*, vol. 732, pp. 87, 2023."
    
    [76] "Xuan-Ha Nguyen, Kim-Hung Le, *Robust detection of unknown DoS/DDoS attacks in IoT networks using a hybrid learning model*, in *Internet of Things*, pp. 100851, 2023."
    
    [77] "Ala Mughaid et al., *Utilizing Machine Learning Algorithms for Effectively Detection IoT DDoS Attacks*, in *Proceedings of the 2023 International Conference on Advances in Computing Research (ACR’23)*, vol. 700, pp. 617, 2023."
    
    [78] "Siamak Layeghy et al., *DI-NIDS: Domain invariant network intrusion detection system*, in *Knowledge-Based Systems*, pp. 110626, 2023."
    
    [79] "Fahad Mazaed Alotaibi, *Network Intrusion Detection Model Using Fused Machine Learning Technique*, in *Computers, Materials & Continua*, vol. 75, no. 2, pp. 2479, 2023."
    
    [80] "Hao Zhang et al., *Improve the Security of Industrial Control System: A Fine-Grained Classification Method for DoS Attacks on Modbus/TCP*, in *Mobile Networks and Applications*, 2023."
    
    [81] "Gustavo de Carvalho Bertoli et al., *Generalizing intrusion detection for heterogeneous networks: A stacked-unsupervised federated learning approach*, in *Computers & Security*, vol. 127, pp. 103106, 2023."
    
    [82] "Shalaka Mahadik et al., *Efficient Intelligent Intrusion Detection System for Heterogeneous Internet of Things (HetIoT)*, in *Journal of Network and Systems Management*, vol. 31, no. 1, 2023."
    
    [83] "Giovanni Apruzzese et al., *The Role of Machine Learning in Cybersecurity*, in *Digital Threats: Research and Practice*, vol. 4, no. 1, pp. 1, 2023."
    
    [84] "Daniyal M. Alghazzawi et al., *Optimized Generative Adversarial Networks for Adversarial Sample Generation*, in *Computers, Materials & Continua*, vol. 72, no. 2, pp. 3877, 2022."
    
    [85] "Yanan Li et al., *HDFEF: A hierarchical and dynamic feature extraction framework for intrusion detection systems*, in *Computers & Security*, vol. 121, pp. 102842, 2022."
    
    [86] "Haider al-Khateeb et al., *Pivot Attack Classification for Cyber Threat Intelligence*, in *Journal of Information Security and Cybercrimes Research*, vol. 5, no. 2, pp. 91, 2022."
    
    [87] "Murad Ali Khan et al., *An optimized ensemble prediction model using AutoML based on soft voting classifier for network intrusion detection*, in *Journal of Network and Computer Applications*, pp. 103560, 2022."
    
    [88] "Loc Gia Nguyen, Kohei Watabe, *Flow-based network intrusion detection based on BERT masked language model*, in *Proceedings of the 3rd International CoNEXT Student Workshop*, pp. 7, 2022."
    
    [89] "Jinghong Lan et al., *A novel hierarchical attention-based triplet network with unsupervised domain adaptation for network intrusion detection*, in *Applied Intelligence*, 2022."
    
    [90] "Raj Kumar Batchu, Hari Seetha, *On improving the performance of DDoS attack detection system*, in *Microprocessors and Microsystems*, vol. 93, pp. 104571, 2022."
    
    [91] "Jyoti Verma et al., *iNIDS: SWOT Analysis and TOWS Inferences of State-of-the-Art NIDS solutions for the development of Intelligent Network Intrusion Detection System*, in *Computer Communications*, vol. 195, pp. 227, 2022."
    
    [92] "Devrim Akgun et al., *A new DDoS attacks intrusion detection model based on deep learning for cybersecurity*, in *Computers & Security*, vol. 118, pp. 102748, 2022."
    
    [93] "Arjun Singh et al., *Intrusion Detection System Using Deep Learning Asymmetric Autoencoder (DLAA)*, in *International Journal of Fuzzy System Applications*, vol. 11, no. 2, pp. 1, 2022."
    
    [94] "Jevgenijus Toldinas et al., *Framing Network Flow for Anomaly Detection Using Image Recognition and Federated Learning*, in *Electronics*, vol. 11, no. 19, pp. 3138, 2022."
    
    [95] "Jinbu Geng et al., *HNOP: Attack Traffic Detection Based on Hierarchical Node Hopping Features of Packets*, in *Computational Science ? ICCS 2022*, vol. 13350, pp. 431, 2022."
    
    [96] "Eric Gyamfi, Anca Jurcut, *Intrusion Detection in Internet of Things Systems: A Review on Design Approaches Leveraging Multi-Access Edge Computing, Machine Learning, and Datasets*, in *Sensors*, vol. 22, no. 10, pp. 3744, 2022."
    
    [97] "Jieling Li et al., *Semi-supervised machine learning framework for network intrusion detection*, in *The Journal of Supercomputing*, vol. 78, no. 11, pp. 13122, 2022."
    
    [98] "Yuan-Cheng Lai et al., *Machine learning based intrusion detection as a service*, in *Proceedings of the 14th IEEE/ACM International Conference on Utility and Cloud Computing Companion*, pp. 1, 2021."
    
    [99] "Mohamed Amine Ferrag et al., *Deep Learning-Based Intrusion Detection for Distributed Denial of Service Attack in Agriculture 4.0*, in *Electronics*, vol. 10, no. 11, pp. 1257, 2021."
    
