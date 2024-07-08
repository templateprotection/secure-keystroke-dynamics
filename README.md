# Overview
This repository contains the code and results for our research on the feasibility of biometric cryptosystems applied to keystroke dynamics, a behavioral biometric modality. More specifically, we apply Dr. Benjamin Fuller's Fuzzy Extraction scheme (https://github.com/benjaminfuller/CompFE/blob/main/PythonImpl/FuzzyExtractor.py) to keystroke dynamics using the Aalto Dataset and the TypeNet model. Our findings demonstrate the potential for secure and practical biometric template protection.

# Introduction / Abstract
The increased use of biometrics for authentication drives the need for sensitive data to be stored in a secure manner. **Biometric Cryptosystems** (i.e., Fuzzy Extractor, Fuzzy Vault) offer a unique solution to accurately compare biometric samples without compromising user security. Current applications of biometric cryptosystems focus heavily on physiological modalities such as face or iris. To overcome this, we review biometric cryptosystems that are applied to **Keystroke Dynamics**, a behavioral modality which identifies the unique typing patterns of individuals, and find that many implementations are lacking in either practicality, comparability, or reproducibility. Thus, we perform a practical application Dr. Benjamin Fuller's Fuzzy Extraction scheme to Keystroke Dynamics, and report the results in a comparable manner. We use the **Aalto Dataset** and **TypeNet** model (https://github.com/BiDAlab/TypeNet/tree/main), a siamese neural network that produces embeddings which can be compared via Euclidean distance for similarity. The unprotected model achieves a 2.37% EER for 1 embedding, and 0.85\% EER fusing 5 embeddings. After applying fuzzy extraction, it achieves 11.09\% EER for 1 embedding and 3.19\% fusing 5 embeddings, demonstrating authentication feasibility.

# Files
- **quantization_loss.py**: This script loads all embeddings and compares their euclidean distances to their quantized hamming distances. The output prints the equal error rates for both modes, as well as creates two score distribution plots to visualize the separability of genuine embeddings and imposter embeddings, before and after quantization.
- **authentication_simulator.py**: This script loads all embeddings and performs authentication on each user using the FuzzyExtractor module. The result of each authentication (success or failure, genuine or imposter) is tracked, and overall metrics are calculated to judge the performance of the system.
- **FuzzyExtractor.py**: An optimization of Dr. Benjamin Fuller's fuzzy extractor module, enhanced with numpy vectorization.
- **FuzzyExtractorMock.py**: A trimmed, **unsafe** version of the FuzzyExtractor module which does not hash each locker. This is used solely for performance evaluation, as it provides authentication results at a much greater speed, but with zero security. 
- **QuantizationUtils.py**: A utility module that contains common code between quantization_loss and keystroke_runner. Most importantly is the EmbeddingNormalizer, which is used to binarize embeddings. Other utility functions are in here as well, such as calculating equal error rate (EER), hamming distance between two binary vectors, and the random selection operations.
- **embeddings_gen_1k.npy**: A numpy file which stores the embeddings of 1000 users, in the shape of 1000x15x128:
  - 1000: The number of distinct users
  - 15: The number of embeddings per user
  - 128: The size of each embedding


# Results
Unprotected Model:
2.37% EER for 1 embedding
0.85% EER for 5 embeddings

Protected Model:
11.29% EER for 1 embedding
3.19% EER for 5 embeddings

# Contact
For questions or further information, please contact us at email@example.com (Placeholder pending blind study review)
