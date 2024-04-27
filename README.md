# WomenCode
Live at https://huggingface.co/spaces/SaiSamyuktaPalle/WomenCode

Recording - Explaining the journey & brief demo - https://duke.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=8a7a38dd-1fe7-4c7b-b004-b15f000b3d55
(Attached the Presentation slides)

## Introduction
### Problem Statement:
Menstrual cycles present various challenges for individuals, including pains, mood swings, and energy fluctuations. Research indicates that proper nutrition and regular exercise can alleviate these issues and contribute to overall well-being during menstrual cycles. However, there is a notable absence of accessible and personalized guidance on nutrition and exercise plans tailored to the distinct phases of the menstrual cycle.
Understanding the nuances of nutrition and exercise requirements across the four phases of the menstrual cycle is essential for individuals to optimize their health and well-being. Yet, existing resources often lack comprehensive information on which nutrients and exercises are most beneficial for each phase. As a result, individuals are unable to effectively manage their menstrual cycle-related symptoms and may not fully understand their bodies' needs throughout the menstrual cycle.

### Prior Efforts
1. FitrWoman
2. Wild.ai
They both offer period tracking and recommends based on the cycle. However, they do not accomodate to dietary needs like vegan.

## Important Directory
1. `notebooks/` - Contains data-preprocessing, TF-IDF & BART FineTuning and Inference scripts
2. `data/output` - Contains the inference (Human evaluation results)
3. `scripts/` - Indexing for RAG & Retrieval of RAG or recipe based on user_input scripts.
4. `app.py` - Integrates all the files & contains the Streamlit UI

### Tree Structure
├───app.py
├───.env
├───data
│   ├───output
│   │       inference.csv
│   │
│   ├───processed
│   │       chunks.json
│   │       inst-resp.zip
│   │       test_data.csv
│   │
│   └───raw
│           womancode-alisa-vitti.pdf
│
├───notebooks
│       data-processing.ipynb
│       dl-and-naive.ipynb
│       inference.ipynb
│       non_dl.ipynb
│
└───scripts
    │   indexing.py
    │   retrieval.py
├───requirements.txt


## Evaluation
Evaluated the models on BLEU Score for Quantitative analysis & Human Evaluation for Qualitative analysis.
Fine Tuned BART for Recipe Generation performs the best with BLEU Score of 0.3317 & Human Evaluation Score of 4.0526

Thank you!