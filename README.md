# kf-pls
 Kernel PLS optimized with Kernel Flows

The present code repository illustrates the method present in:

Zina-Sabrina Duma, Jouni Susiluoto, Otto Lamminpää, Tuomas Sihvonen, Satu-Pia Reinikainen, Heikki Haario,
KF-PLS: Optimizing Kernel Partial Least-Squares (K-PLS) with Kernel Flows,
Chemometrics and Intelligent Laboratory Systems,
2024,
105238,
ISSN 0169-7439,
https://doi.org/10.1016/j.chemolab.2024.105238.
(https://www.sciencedirect.com/science/article/pii/S0169743924001783)

Abstract: Partial Least-Squares (PLS) regression is a widely used tool in chemometrics for performing multivariate regression. As PLS has a limited capacity of modelling non-linear relations between the predictor variables and the response, Kernel PLS (K-PLS) has been introduced for modelling non-linear predictor-response relations. Most available studies use fixed kernel parameters, reducing the performance potential of the method. Only a few studies have been conducted on optimizing the kernel parameters for K-PLS. In this article, we propose a methodology for the kernel function optimization based on Kernel Flows (KF), a technique developed for Gaussian Process Regression (GPR). The results are illustrated with four case studies. The case studies represent both numerical examples and real data used in classification and regression tasks. K-PLS optimized with KF, called KF-PLS in this study, is shown to yield good results in all illustrated scenarios, outperforming literature results and other non-linear regression methodologies. In the present study, KF-PLS has been compared to convolutional neural networks (CNN), random trees, ensemble methods, support vector machines (SVM), and GPR, and it has proved to perform very well.

Keywords: Kernel Partial Least-Squares; Hyperparameter learning; Kernel Flows; Non-linear regression

The code has two implementations: one in MATLAB, and one in Julia. The code is illustrated with the Concrete Strength dataset of "Modeling of strength of high-performance concrete using artificial neural networks" by I. Yeh. 1998, Published in Cement and Concrete Research, Vol. 28, No. 12. Data source: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength.

The method is also utilized in:

Zina-Sabrina Duma, Tomas Zemcik, Simon Bilik, Tuomas Sihvonen, Peter Honec, Satu-Pia Reinikainen, Karel Horak,
Varroa destructor detection on honey bees using hyperspectral imagery,
Computers and Electronics in Agriculture, Volume 224, 2024, 109219, ISSN 0168-1699,
https://doi.org/10.1016/j.compag.2024.109219. (https://www.sciencedirect.com/science/article/pii/S0168169924006100)
