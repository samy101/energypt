# Code for the paper "Towards Pre-trained Models for Load Forecasting"

The emergence of pre-trained models has transformed many fields, including computer vision, natural language processing, and speech recognition, by enabling robust, task-agnostic representations that generalize well across diverse downstream tasks. We present three pre-trained models for load forecasting. The proposed models include Energy-TTMs (adapted from IBM's Tiny Time Mixers), W-LSTMix, and MixForecast. These models are pre-trained on a large-scale energy metering dataset, with 1.26 billion readings, collected from 76,217 real buildings spanning multiple regions, building types, and usage scenarios. This extensive training enables the models to capture complex temporal patterns across diverse building types and operational scenarios. We benchmark the performance of our pre-trained models against six recent time series foundation models (TSFMs), such as Chronos, Lag-Llama, Moirai, TimesFM, TTMs, and MOMENT, as well as multiple traditional and machine learning-based forecasting models, under zero-shot and transfer learning settings, on a large-scale real-world dataset of over 1,767 residential and commercial buildings for the task of short-term load forecasting (STLF). Our results show our pre-trained models can outperform task-specific models in zero-shot settings, highlighting their generalizability and versatility. Finally, we share insights to guide future development of pre-training models for energy data analytics.

---


## Repository Structure
```
energypt/
├── dataset_in/ # In-distribution datasets
├── dataset_out/ # Out-of-distribution datasets
├── MixForecast/ # MixForecast model implementation, experiments, and results
├── energy-ttm/ # Energy-TTMs model implementation, experiments, and results
└── w-lstmix/ # W-LSTMix model implementation, experiments, and results
```

Each model folder includes the **code, pre-trained weights, and results** for reproducibility.


# Pre-trained Models for Load Forecasting

## Generic Pre-trained Models

We consider several state-of-the-art pre-trained models (see Table 1) designed for generic (multi-domain) time series analysis. These models establish the foundation for transferability across diverse domains, including energy.  

#### Table 1. Comparison of pre-trained models for generic (multi-domain) time series  

| Model        | Architecture     | Parameters | Maximum Context Length  | Downstream Tasks                                                | Pre-training Data | Energy Data Used |
|--------------|------------------|------------|------:|----------------------------------------------------------------|-------------------|------------------|
| Chronos      | Encoder-Decoder  | 46M        | 512  | Forecasting                                                    | 893K Series       | UCIE (370)       |
| Moirai       | Encoder-only     | 14M        | 5000 | Forecasting                                                    | 27B Obs           | Buildings-900K   |
| TimesFM      | Decoder-only     | 200M       | 512  | Forecasting                                                    | 100B Obs          | UCIE (370)       |
| Lag-Llama    | Decoder-only     | 2.45M      | 1024 | Forecasting                                                    | 1B Obs            | LCL, UCIE (370)  |
| TTMs         | MLP              | 1M         | 1536 | Forecasting                                                    | 282K Series       | LCL, AUE         |
| TSPulse      | MLP              | 1M         | 512  | Imputation, Anomaly Detection, Classification, Similarity Search | 1B Obs          | -                |
| MOMENT       | Encoder-only     | 125M       | 512  | Forecasting, Imputation, Anomaly Detection, Classification     | 1.13B Obs         | UCIE (370)       |


## Energy Pre-trained Models

In this paper, we consider the following four models (see Table 2) to compare their load forecasting performance. Except for Transformer-L (Gaussian), all models are pre-trained on our dataset containing **1.26 billion readings from 76,217 real buildings**.

#### Table 2. Comparison of pre-trained models for energy time series  

| Model                   | Architecture     | Parameters | Maximum Context Length | Downstream Task   | Pre-training Dataset                     |
|--------------------------|------------------|------------|-----------------------:|--------------------|------------------------------------------|
| Transformer-L (Gaussian) | Encoder-Decoder  | 160M       | 168                   | Load Forecasting        | Buildings-900K (7.9B points)       |
| **MixForecast (Ours)**   | **MLP**          | **0.69M**  | 168                   | **Load forecasting** | **Our dataset (1.26B readings)** |
| **W-LSTMix (Ours)**      | **MLP, LSTM**    | **0.13M**  | 168                   | **Load forecasting** | **Our dataset (1.26B readings)** |
| **Energy-TTMs (Ours)**   | **MLP**          | **1M**     | 168                   | **Load forecasting** | **Our dataset (1.26B readings)** |


### Model Descriptions

- **Transformer-L (Gaussian) from [BuildingsBench](https://nrel.github.io/BuildingsBench/)**  
  Based on the Transformer architecture and pre-trained on simulated hourly electricity data from over 900,000 buildings. It uses a fixed context length of 168 and outputs Gaussian distribution parameters for probabilistic forecasting.

- **MixForecast**  
  A hybrid model that enhances the N-BEATS framework by integrating TSMixer blocks. It captures complex temporal dependencies and multivariate interactions. 

- **W-LSTMix**  
  A modular hybrid model combining LSTM and MLP-Mixer blocks. It decomposes time series into trend and seasonal-residual components, modeling each with specialized stacks. 

- **Energy-TTMs**  
  An adaptation of the Tiny Time Mixers (TTMs) architecture. This lightweight MLP-based model excels in zero-shot and few-shot scenarios. It supports multivariate forecasting and exogenous variables, and is efficient to train. We pre-train this version on our dataset and refer to it as Energy-TTMs.



**Important:** While our models can be pre-trained with higher context lengths, in this study we limit the maximum context length to **168**. This decision aligns with the short-term load forecasting objective (predicting the next 24 hours using the past 168 hours of hourly data) and ensures a fair comparison with the existing Transformer-L (Gaussian) model from the BuildingsBench platform.


---

## Dataset, Model Pre-Training, and Evaluation Settings

### Dataset Summary

| Type                   | # Buildings | # Data Points |
|-------------------------|-------------|---------------|
| **Pre-training**        |             |               |
| Commercial Buildings    | 2,792       | 59M           |
| Residential Buildings   | 73,425      | 1.2B          |
| **Total**               | **76,217**  | **1.26B**     |
|                         |             |               |
| **Evaluation - Out-of-Distribution (OOD)** |             |               |
| Commercial Buildings    | 253         | 2.32M         |
| Residential Buildings   | 245         | 2.37M         |
| **Total**               | **498**     | **4.69M**     |
|                         |             |               |
| **Evaluation - In-Distribution (ID)** |             |               |
| Commercial Buildings    | 98          | 1.56M         |
| Residential Buildings   | 1,171       | 16M           |
| **Total**               | **1,269**   | **17.56M**    |


### Model Pre-Training

Energy-TTMs, MixForecast, and W-LSTMix are pre-trained on energy consumption data from **76,217 buildings** using an 8-day (192-hour) sliding window with a 1-day stride. 

- **Energy-TTMs:**  
  Encoder with 3 TSMixer blocks (\(d\_model=16\)), GELU activations, gated channel-mixing (no self-attention), dropout 0.3.  
  Decoder with 8 TSMixer blocks (\(d\_model=8\)), trained up to 100 epochs on 4×H100 GPUs (~40 hrs).  

- **MixForecast:**  
  Hierarchical architecture with 6 stacks × 3 TSMixer blocks (hidden dim 512), expansion factors [4,2,1], GELU activations, Huber loss.  
  Trained up to 35 epochs on 4×V100 GPUs (~4 days).  

- **W-LSTMix:**  
  Decomposes input into trend + seasonal components. Each stack has 3 hybrid blocks (hidden dim 256, patch size 8).  
  Uses Huber loss, trained up to 35 epochs on 4×AMD MI300X GPUs (~4 days).  

⚡ **Note:** All our models are non-transformer based, so they do **not require GPUs for inference or fine-tuning**, unlike general-purpose TSFMs.  

### Evaluation Protocol and Baseline Models

We consider the following categories of models, following the evaluation protocols from the [BuildingsBench](https://nrel.github.io/BuildingsBench/). 

- **Not Pre-trained + Not Fine-tuned**  
  Models that are used without any prior training or task-specific adaptation. Typically simple baselines.  
  **Example**: `Naive (Mean)`

- **Not Pre-trained + Fine-tuned**  
  Models trained from scratch using only the target dataset. They do not leverage any external or prior knowledge.  
  **Examples**: `Auto-ARIMA`, `LR`, `LightGBM`, `PatchTST`, `TFT`

- **Pre-trained + Not Fine-tuned**  
  Models trained on large, general datasets and directly applied to new tasks without further tuning (zero-shot).  
  **Examples**: `Moirai`, `Lag-Llama`, `Chronos`, `TimesFM`, `MOMENT`, `TTMs`

- **Pre-trained (Energy) + Not Fine-tuned**  
  Domain-specific models pre-trained on large-scale energy datasets and used directly without task-specific tuning.  
  **Examples**: `Transformer-L (G)`, `Energy-TTMs (Ours)`, `Mix-Forecast (Ours)`, `W-LSTMix (Ours)`

- **Pre-trained + Fine-tuned**  
  Models first trained on large, general datasets and then fine-tuned on the target dataset for improved performance.  
  **Examples**: `Moirai`, `Lag-Llama`, `MOMENT`



---


## Results (Updated)

```diff
✅ 
+ Note: These are our final results after fixing an error.
+ Due to a technical error, the submitted version contains incorrect results for our Energy-TTMs model.
+ The correct results can be verified in the respective model folders, which include both the reproducible code and the results for each dataset.
```

The following tables compare model performance:

- **Table 3:** Summary of out-of-distribution dataset results using mean **NRMSE**.  
- **Tables 4 and 5:** In-distribution dataset results (**1,269 buildings from 23 datasets**).  
- **Tables 6 and 7:** Out-of-distribution dataset results (**498 buildings from 8 datasets**).  


#### Table 3: Comparison of forecasting error using mean NRMSE across all out-of-distribution datasets


| **Models**             | **Commercial** | **Residential** |
|------------------------|----------------|-----------------|
| **Not Pre-trained + Not Fine-tuned** |                |                 |
| Naive (Mean)           | 47.72          | 96.86           |
| **Not Pre-trained + Fine-tuned** |                |                 |
| Auto-ARIMA              | 48.61          | _85.00_         |
| LR                     | _37.51_        | 103.47          |
| LightGBM               | **35.26**      | 110.37          |
| PatchTST               | 36.54          | **71.73**       |
| TFT                    | 39.81          | 76.76           |
| **Pre-trained + Not Fine-tuned** |                |                 |
| Moirai                 | 53.10          | 76.14           |
| Lag-Llama             | 57.45          | 82.49           |
| Chronos               | _33.87_        | 73.19           |
| TimesFM               | 48.67          | **66.98**       |
| MOMENT                | 44.91          | 76.37           |
| TTMs                  | **30.90**      | _73.16_         |
| **Pre-trained (Energy) + Not Fine-tuned** |         |                 |
| Transformer-L (G)      | -              | 85.67           |
| Energy-TTMs (Ours)     | _17.41_        | _76.75_         |
| Mix-Forecast (Ours)    | 20.57          | 83.64           |
| W-LSTMix (Ours)        | **12.74**      | **49.94**       |
| **Pre-trained + Fine-tuned** |                |                 |
| Moirai                 | **27.24**      | **57.18**       |
| Lag-Llama             | _30.92_        | _66.63_         |
| MOMENT                | 42.95          | 73.20           |


### In-Distribution Dataset Results

- This evaluation set is derived from datasets where a subset of buildings is used during pre-training, while the remaining buildings are held out for evaluation. 
- This split contains 1,269 buildings, covering both commercial and residential types.
  
#### Table 4: Comparison of forecasting error using median NRMSE on in-distribution datasets - Traditional ML models and Energy Pre-trained Models

**Commercial Buildings**

| Dataset   | Naive | Auto-ARIMA | LR    | LightGBM | PatchTST | TFT   | Energy-TTMs | MixForecast | W-LSTMix |
|-----------|-------|------------|-------|-----------|----------|-------|--------------|--------------|-----------|
| BDG-2     | 28.95 | 29.05      | 21.65 | 21.71     | 21.24    | 26.06 | 14.08        | 14.49        | 10.16     |
| Enernoc   | 43.56 | 40.80      | 26.29 | 26.05     | 28.41    | 33.60 | 24.35        | 19.15        | 18.98     |
| IBlend    | 36.65 | 37.85      | 21.44 | 22.15     | 22.06    | 21.05 | 32.45        | 33.49        | 21.57     |
| PSS       | 107.90| 113.82     | 76.13 | 72.34     | 79.73    | 94.24 | 69.34        | 65.77        | 60.44     |
| SKC       | 41.15 | 40.47      | 45.68 | 39.37     | 48.49    | 43.76 | 35.12        | 33.46        | 24.50     |
| UNICON    | 28.10 | 29.65      | 33.84 | 29.95     | 19.31    | 20.15 | 17.24        | 21.46        | 14.11     |

**Residential Buildings**

| Dataset   | Naive | Auto-ARIMA | LR    | LightGBM | PatchTST | TFT   | Energy-TTMs | MixForecast | W-LSTMix |
|-----------|-------|------------|-------|-----------|----------|-------|--------------|--------------|-----------|
| DESM      | 82.33 | 84.48      | 78.68 | 70.15     | 58.39    | 64.52 | 65.37        | 65.90        | 59.64     |
| DTH       | 104.20| 95.11      | 34.44 | 34.05     | 30.46    | 81.04 | 29.32        | 27.11        | 22.59     |
| ECCC      | 104.10| 86.67      | 87.07 | 83.76     | 72.46    | 83.27 | 62.37        | 60.43        | 59.25     |
| GoiEner   | 143.39| 126.59     | 131.40| 129.35    | 140.30   | 119.48| 84.81        | 169.08       | 50.92     |
| HES       | 105.09| 91.63      | 111.20| 128.14    | 90.68    | 96.35 | 57.10        | 222.48       | 42.65     |
| HSG       | 0.94  | 0.94       | 32.44 | 54.59     | 4.38     | 1.47  | 6.28         | 6.54         | 2.01      |
| HUE       | 91.28 | 90.77      | 249.94| 289.03    | 75.46    | 75.80 | 45.91        | 119.15       | 34.54     |
| IRH       | 100.49| 98.17      | 95.09 | 92.15     | 81.52    | 87.62 | 80.98        | 80.78        | 74.39     |
| NEEA      | 110.49| 87.80      | 93.59 | 91.42     | 76.12    | 80.39 | 93.27        | 97.05        | 60.70     |
| NESEMP    | 158.39| 132.68     | 123.04| 123.56    | 76.23    | 89.24 | 74.78        | 121.76       | 54.37     |
| Norwegian | 63.78 | 59.06      | 62.28 | 61.15     | 54.36    | 55.40 | 45.44        | 45.41        | 43.01     |
| PES       | 109.11| 87.70      | 196.68| 204.44    | 62.55    | 87.45 | 38.75        | 90.51        | 39.75     |
| RSL       | 99.11 | 84.60      | 111.19| 138.36    | 79.20    | 83.20 | 46.85        | 113.03       | 32.24     |
| SAVE      | 1.01  | 1.06       | 8.13  | 40.67     | 4.13     | 0.93  | 1.53         | 6.35         | 1.95      |
| SGSC      | 129.07| 112.43     | 115.05| 112.57    | 97.19    | 104.68| 86.03        | 113.02       | 50.80     |
| UKST      | 110.87| 98.13      | 99.91 | 102.13    | 89.58    | 94.45 | 64.40        | 93.70        | 51.15     |
| iFlex     | 41.92 | 39.61      | 41.91 | 40.32     | 34.38    | 34.78 | 16.87        | 58.95        | 14.69     |


### Table 5: Comparison of forecasting error using median NRMSE on in-distribution datasets – TSFM and Pre-trained Models
             

**Commercial Buildings**
| Dataset   | Moirai | Lag-Llama | Chronos | TimesFM | Moment | TTMs  | TF-L (G) | Moirai (FT) | Lag-Llama (FT) | Moment (FT) | W-LSTMix |
|-----------|--------|-----------|---------|---------|--------|-------|----------|--------------|----------------|--------------|-----------|
| BDG-2     | 29.49  | 46.37     | 14.56   | 29.05   | 28.31  | 15.45 | 103.55   | 15.18        | 18.57          | 29.08        | 10.16     |
| Enernoc   | 28.97  | 43.61     | 21.92   | 40.80   | 41.01  | 21.17 | 109.56   | 22.87        | 30.36          | 35.95        | 18.98     |
| IBlend    | 75.10  | 55.13     | 30.10   | 37.85   | 35.85  | 30.31 | 111.03   | 24.90        | 21.44          | 22.44        | 21.57     |
| PSS       | 83.09  | 95.98     | 84.30   | 113.82  | 96.44  | 70.07 | 133.89   | 47.59        | 63.52          | 97.12        | 60.44     |
| SKC       | 37.65  | 46.65     | 36.17   | 40.47   | 42.97  | 31.12 | 110.95   | 32.63        | 31.39          | 49.25        | 24.50     |
| UNICON    | 64.32  | 56.98     | 16.17   | 29.65   | 24.86  | 17.25 | 101.47   | 20.25        | 20.26          | 23.85        | 14.11     |

**Residential Buildings**
| Dataset   | Moirai | Lag-Llama | Chronos | TimesFM | Moment | TTMs  | TF-L (G) | Moirai (FT) | Lag-Llama (FT) | Moment (FT) | W-LSTMix |
|-----------|--------|-----------|---------|---------|--------|-------|----------|--------------|----------------|--------------|-----------|
| DESM      | 80.75  | 80.75     | 71.60   | 66.98   | 74.33  | 63.70 | 130.33   | 50.39        | 62.91          | 63.56        | 59.64     |
| DTH       | 34.93  | 53.28     | 28.66   | 27.68   | 82.49  | 30.17 | 108.25   | 33.22        | 29.37          | 61.16        | 22.59     |
| ECCC      | 69.05  | 78.11     | 76.68   | 64.02   | 79.78  | 61.57 | 98.05    | 64.82        | 72.45          | 83.75        | 59.25     |
| GoiEner   | 121.39 | 131.44    | 121.14  | 111.12  | 119.02 | 117.69| 15.65    | 101.45       | 110.13         | 117.85       | 50.92     |
| HES       | 85.64  | 105.07    | 92.00   | 89.23   | 90.99  | 74.88 | 98.09    | 79.66        | 91.25          | 99.47        | 42.65     |
| HSG       | 1.04   | 2.60      | 1.78    | 2.75    | 2.89   | -     | 152.99   | 2.38         | 3.27           | 0.87         | 2.01      |
| HUE       | 87.45  | 92.74     | 77.05   | 71.76   | 78.24  | -     | 17.64    | 57.78        | 62.26          | 82.05        | 34.54     |
| IRH       | 85.49  | 100.60    | 92.36   | 80.68   | 94.47  | 76.66 | 137.60   | 67.93        | 85.37          | 85.01        | 74.39     |
| NEEA      | 96.10  | 90.14     | 87.56   | 75.49   | 83.80  | 71.73 | 155.39   | 68.33        | 87.45          | 79.48        | 60.70     |
| NESEMP    | 123.64 | 111.69    | 95.87   | 79.26   | 95.04  | -     | 135.66   | 66.13        | 74.74          | 82.41        | 54.37     |
| Norwegian | 49.36  | 62.24     | 52.44   | 52.68   | 51.84  | 46.83 | 28.72    | 44.13        | 53.94          | 54.85        | 43.01     |
| PES       | 68.21  | 81.60     | 75.70   | 62.66   | 79.20  | -     | 22.87    | 47.04        | 56.64          | 67.61        | 39.75     |
| Plegma    | 139.77 | 142.43    | 141.66  | 140.31  | 138.57 | 186.82| 224.11   | 110.99       | 133.61         | 137.34       | 87.23     |
| RSL       | 99.39  | 87.34     | 84.50   | 62.76   | 75.48  | 94.76 | 13.19    | 61.62        | 71.33          | 74.10        | 32.24     |
| SAVE      | 1.12   | 2.91      | 1.81    | 2.75    | 3.37   | 10.86 | 118.38   | 1.03         | 2.36           | 1.73         | 1.95      |
| SGSC      | 97.46  | 109.90    | 101.12  | 91.06   | 99.85  | 78.18 | 33.97    | 78.46        | 90.52          | 100.13       | 50.80     |
| UKST      | 96.69  | 98.72     | 82.79   | 83.05   | 87.24  | 70.03 | 20.86    | 69.61        | 76.42          | 88.11        | 51.15     |
| iFlex     | 32.96  | 53.19     | 32.76   | 41.38   | 38.11  | 40.31 | 29.45    | 24.24        | 35.24          | 38.10        | 14.69     |


### Out-of-Distribution (OOD) Dataset Results

- This evaluation set comprises datasets entirely excluded from the pre-training corpus, enabling assessment of model generalization to unseen buildings across distinct regions.
- This split contains 498 buildings from 8 commercial and residential datasets


#### Table 6: Comparison of forecasting error using median NRMSE on Out-of-distribution (ODD) datasets - Traditional ML models and Energy Pre-trained Models
              
**Commercial Buildings**

| Dataset         | Naive  | Auto-ARIMA | LR     | LightGBM | PatchTST | TFT    | Energy-TTMs | MixForecast | W-LSTMix |
|----------------|--------|------------|--------|-----------|----------|--------|--------------|--------------|-----------|
| IPC-Commercial | 84.53  | 59.13      | 44.20  | 44.83     | 55.61    | 54.68  | 29.49        | 28.94        | 13.41     |
| NREL           | 39.17  | 43.10      | 12.05  | 12.50     | 21.01    | 30.10  | 16.31        | 12.19        | 12.06     |

**Residential Buildings**

| Dataset         | Naive  | Auto-ARIMA | LR     | LightGBM | PatchTST | TFT    | Energy-TTMs | MixForecast | W-LSTMix |
|----------------|--------|------------|--------|-----------|----------|--------|--------------|--------------|-----------|
| CEEW           | 100.74 | 84.04      | 129.24 | 132.07    | 89.82    | 90.83  | 62.12        | 89.06        | 49.81     |
| ECWM           | 48.69  | 52.03      | 53.52  | 50.39     | 38.00    | 38.34  | 37.50        | 38.30        | 32.42     |
| HONDA-SH       | 33.01  | 37.28      | 15.33  | 14.96     | 26.22    | 25.79  | 12.64        | 12.83        | 11.30     |
| RHC            | 76.08  | 71.02      | 66.23  | 64.33     | 56.94    | 57.79  | 51.54        | 51.13        | 50.21     |
| NREL           | 84.68  | 78.65      | 49.83  | 49.62     | 52.92    | 76.32  | 42.15        | 40.53        | 38.97     |
| fIEECe         | 85.84  | 63.18      | 67.37  | 64.13     | 53.92    | 67.26  | 41.14        | 49.73        | 44.47     |


#### Table 7: Comparison of forecasting error using median NRMSE on Out-of-distribution (OOD) datasets – TSFM and Pre-trained Models

**Commercial Buildings**

| Dataset         | Moirai | Lag-Llama | Chronos | TimesFM | Moment | TTMs  | TF-L (G) | Moirai (FT) | Lag-Llama (FT) | Moment (FT) | W-LSTMix |
|----------------|--------|-----------|---------|---------|--------|-------|----------|--------------|----------------|--------------|-----------|
| IPC-Commercial | 102.07 | 53.94     | 24.69   | 27.48   | 68.48  | 32.38 | 122.50   | 42.05        | 39.46          | 41.48        | 13.41     |
| NREL           | 21.25  | 42.57     | 16.90   | 20.31   | 35.61  | 12.30 | 101.75   | 13.62        | 19.62          | 35.10        | 12.06     |

**Residential Buildings**

| Dataset         | Moirai | Lag-Llama | Chronos | TimesFM | Moment | TTMs  | TF-L (G) | Moirai (FT) | Lag-Llama (FT) | Moment (FT) | W-LSTMix |
|----------------|--------|-----------|---------|---------|--------|-------|----------|--------------|----------------|--------------|-----------|
| CEEW           | 266.44 | 87.57     | 95.87   | 60.06   | 76.67  | 88.73 | 16.12    | 96.20        | 75.39          | 81.44        | 49.81     |
| ECWM           | 38.85  | 64.63     | 39.48   | 37.70   | 43.08  | 37.41 | 105.81   | 30.80        | 39.30          | 43.06        | 32.42     |
| HONDA-SH       | 14.00  | 39.28     | 12.80   | 13.66   | 26.30  | 13.02 | 102.90   | 11.29        | 16.97          | 24.50        | 11.30     |
| RHC            | 56.52  | 64.55     | 59.83   | 82.84   | 60.91  | 52.05 | 116.68   | 52.14        | 54.92          | 55.10        | 50.21     |
| NREL           | 44.57  | 65.15     | 43.15   | 54.01   | 65.83  | 40.70 | 19.54    | 33.80        | 44.34          | 62.08        | 38.97     |
| fIEECe         | 53.16  | 64.09     | 55.91   | 63.58   | 57.02  | 49.61 | 13.20    | 40.73        | 54.44          | 54.81        | 44.47     |
