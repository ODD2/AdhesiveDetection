# Adhesive Detection
## Description
This repository include the work completed during internship with the K&S for detecting adhesives on the PCBs. In this work, we tackle the problem of scarce training data by synthesizing the adhesive given only the PCB image. We simulate the reflection and refraction caused by the material of the adhesive to generate the corresponding visual effect to better align with real world datas.

## Create Environment 
```bash
conda env create -f ./environment.yml
```
## Dataset Structure
- Currently, the dataset requires the adhesive mask to the corresponding PCB image in order to rely on both real and synthesized sample during training.
```
./dataset
    |- test/
        |- type_n/
            |- aa_1.bmp
            |- aa_mask.bmp # adhesive mask of aa_1.bmp
            |- bb_1.bmp
            |- bb_mask.bmp # adhesive mask of bb_1.bmp
        ...
    | - train/
        |- type_n/
            |- cc_1.bmp
            |- cc_mask.bmp # adhesive mask of cc_1.bmp
            |- dd_1.bmp
            |- dd_mask.bmp # adhesive mask of dd_1.bmp
        ...
```

# Train
- train a model to segmentate the adhesive on the PCB image.
- the model will be saved at `./models`.
- please refer to `./src/synthesizer.py` to better understand the synthesis process.
 ```shell
python -m main \
--batch_size=70 \
--num_workers=8 \
--epoch=30 \
--lr=5e-5
 ```

# Evaluation
- evaluate the pre-trained model on the validation and test subset with metrics including: accuracy, F1 and IoU
 ```shell
python -m evaluations \
--model_path=./models/xxxx.py
 ```
# Inference
- inference the given image for the adhesive mask, the result will be save to `./mask.png`
 ```shell
python -m inference \
--model_path=./models/xxxx.py \
--image_path=./xxxx.png
 ```


    