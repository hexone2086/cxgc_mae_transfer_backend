# Set the path to save video
OUTPUT_DIR='/home/hanshengliang/mae_deploy/output'
# path to video for visualization
IMAGE_PATH='/home/hanshengliang/WechatIMG12569.jpg'
# path to pretrain model
MODEL_PATH='/home/hanshengliang/mae_deploy/model/checkpoint-1419.pth'
REMAINING_IMG_PATH='/home/hanshengliang/mae_deploy/input/combined_image.npy'
MASK_PATH='/home/hanshengliang/mae_deploy/input/mask_positions.npy'
IMG_MEAN_PATH='/home/hanshengliang/mae_deploy/input/patch_means.npy'
IMG_STD_PATH='/home/hanshengliang/mae_deploy/input/patch_stds.npy'
mask_ratio=0.75

# Now, it only supports pretrained models with normalized pixel targets
python run_mae_vis.py ${IMAGE_PATH} ${OUTPUT_DIR} ${MODEL_PATH} ${REMAINING_IMG_PATH} ${MASK_PATH} ${IMG_MEAN_PATH} ${IMG_STD_PATH} --mask_ratio ${mask_ratio}