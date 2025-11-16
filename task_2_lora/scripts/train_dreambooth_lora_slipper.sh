export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./sample_data/dreambooth-slipper"
export OUTPUT_DIR="./runs/dreambooth_slipper_1000"
export CLASS_DIR="./sample_data/dreambooth-slipper-class"

accelerate launch --mixed_precision="no" train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=$CLASS_DIR \
  --with_prior_preservation \
  --class_prompt="a photo of a slipper" \
  --num_class_images=100 \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of a sks slipper" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --validation_prompt="A photo of sks slipper in a jungle" \
  --validation_epochs=50 \
  --checkpoints_total_limit 2 \
  --seed="0"