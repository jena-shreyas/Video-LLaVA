
# Video-LLaVA

## Instructions

- Follow README.md for instructions to setup the project environment.
- Create a folder `causalvidqa` inside `data` folder. Follow [this link](https://iitkgpacin-my.sharepoint.com/:f:/g/personal/shreyasjena_kgpian_iitkgp_ac_in/Eo0ALiApAlpLjZIaI4O9vhYBImdL9bLX03m_Lc05-aaCiA?e=Luli0u) to download and extract the annotations and videos into `causalvidqa/annotations` and `causalvidqa/videos` respectively.
- Run the following command in the root folder for finetuning on `causalvidqa/annotations/train.json` :


```
bash scripts/v1_5/finetune.sh   # normal finetuning
bash scripts/v1_5/finetune_lora.sh   # LoRA finetuning
```

- Feel free to make changes to the script as per the system requirements.

