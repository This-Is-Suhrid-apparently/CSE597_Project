This project focuses on fine-tuning the Bidirectional Language-Image Pre-training (BLIP) model for the task of Image-to-Text Retrieval. BLIP is a state-of-the-art model that combines the power of natural language understanding and image representation learning. This fine-tuning process aims to enhance BLIP's performance specifically for retrieving relevant textual descriptions given an input image.

I have employed the BLIP with VIT(base) for finetuning on the downstream task of Image to Text Retrieval. 
Limited computing power led me to tweak a few hyper-parameters, as clearly stated in the project report. Primarily the authors have used 8 A100 GPUs to finetune the model whereas I have used just one. To implement this, I have changed some implementation scripts and some hyperparameters in the config files.









### Finetuned checkpoints:
Task | BLIP w/ ViT-B | BLIP w/ ViT-B and CapFilt-L | BLIP w/ ViT-L 
--- | :---: | :---: | :---:
Image-Text Retrieval (COCO) | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth">Download</a>| - | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth">Download</a>
Image-Text Retrieval (Flickr30k) | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_flickr.pth">Download</a>|  - | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_flickr.pth">Download</a>

### Image-Text Retrieval:
1. Download COCO and Flickr30k datasets from the original websites, and set 'image_root' in configs/retrieval_{dataset}.yaml accordingly.
   
2. To evaluate the finetuned BLIP model on Flickr30K, run: 
   <pre>python -m torch.distributed.run --nproc_per_node=1 train_retrieval.py \
   --config ./configs/retrieval_flickr.yaml \
   --output_dir output/retrieval_flickr \</pre>
         
   To evaluate the finetuned BLIP model on COCO, run:
   <pre>python -m torch.distributed.run --nproc_per_node=1 train_retrieval.py \
   --config ./configs/retrieval_coco.yaml \
   --output_dir output/retrieval_coco \
   --evaluate</pre> 

3. To finetune the pre-trained checkpoint on flickr, first set 'pretrained' in configs/retrieval_flickr.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth". Then run:
   <pre>python -m torch.distributed.run --nproc_per_node=1 train_retrieval.py \
   --config ./configs/retrieval_flickr.yaml \
   --output_dir output/retrieval_flickr </pre> 
  
   To finetune the pre-trained checkpoint on coco, first set 'pretrained' in configs/retrieval_coco.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth". Then run:
   <pre>python -m torch.distributed.run --nproc_per_node=1 train_retrieval.py \
   --config ./configs/retrieval_coco.yaml \
   --output_dir output/retrieval_coco </pre> 




### Citation
If you find this code to be useful for your research, please consider citing.
<pre>
@inproceedings{li2022blip,
      title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation}, 
      author={Junnan Li and Dongxu Li and Caiming Xiong and Steven Hoi},
      year={2022},
      booktitle={ICML},
}</pre>

### Acknowledgement
The implementation of BLIP relies on resources from <a href="https://github.com/salesforce/ALBEF">ALBEF</a>, <a href="https://github.com/huggingface/transformers">Huggingface Transformers</a>, and <a href="https://github.com/rwightman/pytorch-image-models/tree/master/timm">timm</a>. We thank the original authors for their open-sourcing.
