---
library_name: transformers
license: apache-2.0
base_model: NourFakih/Vit-GPT2-UCA-UCF-06
tags:
- generated_from_trainer
metrics:
- rouge
model-index:
- name: Vit-GPT2-UCA-UCF-07
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Vit-GPT2-UCA-UCF-07

This model is a fine-tuned version of [NourFakih/Vit-GPT2-UCA-UCF-06](https://huggingface.co/NourFakih/Vit-GPT2-UCA-UCF-06) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1968
- Rouge1: 34.6433
- Rouge2: 13.5351
- Rougel: 29.5099
- Rougelsum: 30.0007
- Gen Len: 16.002

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 16
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss | Rouge1  | Rouge2  | Rougel  | Rougelsum | Gen Len |
|:-------------:|:------:|:----:|:---------------:|:-------:|:-------:|:-------:|:---------:|:-------:|
| 0.4617        | 0.5469 | 500  | 0.1655          | 34.1712 | 12.9219 | 29.0744 | 29.6374   | 16.407  |
| 0.4256        | 1.0930 | 1000 | 0.1755          | 34.2664 | 13.121  | 29.2664 | 29.8242   | 15.724  |
| 0.3498        | 1.6399 | 1500 | 0.1807          | 34.9169 | 13.5342 | 29.5801 | 30.157    | 16.269  |
| 0.3158        | 2.1859 | 2000 | 0.1921          | 33.9586 | 12.8412 | 28.6693 | 29.1732   | 16.157  |
| 0.2768        | 2.7328 | 2500 | 0.1968          | 34.6433 | 13.5351 | 29.5099 | 30.0007   | 16.002  |


### Framework versions

- Transformers 4.47.0
- Pytorch 2.5.1+cu121
- Datasets 3.3.1
- Tokenizers 0.21.0
