# ProtLMsDTA
   Drug-Target affinity (DTA) prediction is an important task for in-silico drug discovery, which could speed up drug development and reduce resource consumption. In recent years, numerous efforts have been made to develop data-hungry algorithms like deep learning methods for predicting DTA. However, some challenges are still open:(i) the amount of labeled DTA data is limited and (ii) the existing protein representation learning approaches in DTA task are hard to extract the useful features on labeled datasets with only hundreds of protein sequences. As such, development of more accurate computational methods for DTA prediction are still highly desirable. Herein, we proposed a deep learning method termed ProtLMsDTA, a predictor using pretrained protein language models (ProtLMs) for predicting DTA. After being pretrained with millions of unlabeled protein sequences, our proposed models could extract the useful features in DTA prediction task according to protein sequences. Then, four successfully pretrained ProtLMs were used for ProtLMsDTA experiments, including ProtT5, ProtBERT, ProtAlbert and ProtXLNet, which were compared among the state-of-the art baselines on Davis and KIBA datasets. Overall, ProtLMsDTA outperforms other methods significantly, representing a powerful tool for the high-throughput and cost effective DTA prediction.

## Requirements
```
python=3.8
rdkit=2020.09.5
pytorch=1.8.2
pytorch_geometric=2.0.1
```
