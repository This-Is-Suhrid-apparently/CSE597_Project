# CSE597_Project
This is the repository for the CSE 597 Final Project


To implement fine-tuning of the BLIP (Bootstrapping Language-Image Pre-training) model with a VIT (Vision Transformer) base for an image-to-text retrieval task, I have followed the following steps:

1) Obtain the COCO and Flickr30k datasets from their respective official websites.
2) Set the 'image_root' parameter in the configuration file (configs/retrieval_{dataset}.yaml) to the root directory where the downloaded images are stored.
3) Evaluate the Finetuned BLIP Model on COCO:
   python -m torch.distributed.run --nproc_per_node=1 train_retrieval.py \
  --config ./configs/retrieval_coco.yaml \
  --output_dir output/retrieval_coco \
  --evaluate
4) Set the 'pretrained' parameter in the configuration file (configs/retrieval_coco.yaml) to the pre-trained checkpoint URL:
   pretrained: "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth"
5) Then run the following script:
   python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py \
  --config ./configs/retrieval_coco.yaml \
  --output_dir output/retrieval_coco 

Due to limited compute power, I could not finetune the model on 8 A100 GPUs as mentioned in the paper. 
I also had to change the training batch size to 8 and the testing batch size to 32 for the implementation to be successful.
For this, I have updated the config files as per requirement.
