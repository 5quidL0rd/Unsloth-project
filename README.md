# Unsloth-Finetuning-Haiku-project
Using UnSloth to Finetune Llama-3 and Use in Ollama

Taking a dataset on haiku downloaded from Kaggle UnSloth was used to fine-tune an ollama model into generating haiku poetry. 

Project interest is directly proportional to the amount of GPU you have access to. 

## **Install Dependencies**

On Google Notebook or whatever you are using for this, install your dependencies. 

```bash
!pip install unsloth
!pip install kaggle

!pip install trl

!pip install peft

!pip install torch
```

UnSloth is a parameter-effecient library designed specifically for fine-tuning LLMs. 
Kaggle is an excellent website for downloading various datasets for projects. 
Finally, torch is an extremely useful library for basically everything ML 


## **Download your Dataset**

Go to Kaggle and download their English-language haiku dataset (or a different dataset if you wish to finetune for other puproses)

https://www.kaggle.com/datasets/hjhalani30/haiku-dataset

To import into your notebook first download it to your PC. Then use the following code to transfer to google collab:

```bash
from google.colab import files

uploaded = files.upload()
```
Select your file and you're ready to roll. 

## **Finding an LLM that won't max out your free GPU consumption**

Using a free version of Google Collab there are different tactics you can use so as to not max out your GPU:
1) Use a small model such as a 1B Llama.
2) reduce your memory utilization
3) reduce your batch size or gradient accumulation steps
4) reduce your training epochs

I recommend using HuggingFace to pull various models to find a free one that works for you. Create an account, input your secret key, and should be good to go. 


## **Further Research**

Using my code you should be able to fine-tune an LLM for improved haiku poem generation. Granted it's a little lame using only free GPUs but it's a good proof of concept. 

For further research: 

UnSloth: https://unsloth.ai

Hugging Face Models: https://huggingface.co/models






