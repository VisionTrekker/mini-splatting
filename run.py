import os

GPU_ID = 3
# scenes = {"building1": "cuda", "town1": "cuda", "block2_sxfx": "cpu", "building2": "cpu", "building3": "cpu"}
# scenes = {"gm_Museum": "cuda"}
scenes = {"anji_qiyu": "cuda"}

# ms
for idx, scene in enumerate(scenes.items()):
    ############ 训练 ############
    print('---------------------------------------------------------------------------------')
    cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
           python ms/train.py \
           -s ../../remote_data/dataset_reality/{scene[0]} \
           -m output/{scene[0]} \
           -r 1 \
           --data_device "{scene[1]}" \
           --port 6020 \
           --test_iterations 7000 15000 30000 \
           --save_iterations 15000 30000 \
           --imp_metric outdoor \
           '
    print(cmd)
    os.system(cmd)
    #
    # ############ 渲染 ############
    # print('---------------------------------------------------------------------------------')
    # cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
    #         python render.py \
    #         -m output/{scene[0]} \
    #         --iteration 30000 \
    #         --skip_test'
    # print(cmd)
    # os.system(cmd)
    #
    # ############ 评测 ############
    # print('---------------------------------------------------------------------------------')
    # cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
    #         python metrics.py \
    #         -m output/{scene[0]}'
    # print(cmd)
    # os.system(cmd)


# ms_d
# for idx, scene in enumerate(scenes.items()):
#     ############ 训练 ############
#     print('---------------------------------------------------------------------------------')
#     cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
#            python ms_d/train.py \
#            -s ../../remote_data/dataset_reality/{scene[0]} \
#            -m output_d/{scene[0]} \
#            -r -1 \
#            --data_device "{scene[1]}" \
#            --port 6019 \
#            --test_iterations 7000 15000 30000 \
#            --save_iterations 15000 30000 \
#            '
#     print(cmd)
#     os.system(cmd)
#
#     ############ 渲染 ############
#     print('---------------------------------------------------------------------------------')
#     cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
#             python render.py \
#             -m output_d/{scene[0]} \
#             --iteration 30000 \
#             --skip_test'
#     print(cmd)
#     os.system(cmd)
#
#     ############ 评测 ############
#     print('---------------------------------------------------------------------------------')
#     cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
#             python metrics.py \
#             -m output_d/{scene[0]}'
#     print(cmd)
#     os.system(cmd)