CUDA_VISIBLE_DEVICES=2 python ./inference.py --config ./configs/inference/inference_pose2img.yaml \
                                           --ref_path ./test_imgs/reference/leverkusen_000013_000019.png\
                                           --seg_path ./test_imgs/seg_map/munster_000038_000019.png \
                                           --output_dir ./outputs \
                                           --cond_mode multi_panop \
                                           --width 256 \
                                           --height 256 \
                                           --seed 42 \ 
                                        
