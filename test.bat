@echo off
setlocal enabledelayedexpansion

:: 定义参数数组
set distance_methods=cos max norm sum
set models=cspresnet wideresnet50 starnets2 starnets4 mobileones0 efficientnet_b7 efficientnet_b0
set datasets=lcz newmo

:: 定义每个数据集对应的resize值
set resize_lcz=519,1372
set resize_newmo=400,400
set layer_wideresnet50=layer1
set layer_starnets2=stages.0
set layer_starnets4=stages.0
set layer_mobileones0=stages.0
set layer_efficientnet_b7=blocks.1
set layer_efficientnet_b0=blocks.1
set layer_cspresnet=stages.0

:: 循环遍历所有参数组合
for %%a in (%distance_methods%) do (
    for %%b in (%models%) do (
        for %%c in (%datasets%) do (
            :: 根据当前数据集选择resize值
            set resize_value=!resize_%%c!
            set layer_value=!layer_%%b!
            echo Running with distance_method=%%a, model=%%b, dataset=%%c, resize=!resize_value!, layer=!layer_value!
            c:\Users\bonozhou\.conda\envs\anomalib_env\python.exe "D:\hust\MediaLab\IAD\Anomalib\patchcore-inspection\bin\run_patchcore.py" --save_patchcore_model --save_segmentation_images --log_group "%%b_!layer_value!" --log_project "%%c_%%a" results "patch_core" -b "%%b" -le "!layer_value!" --target_embed_dimension "128" --anomaly_scorer_num_nn "1" --distance_method "%%a" sampler -p "0.2" approx_greedy_coreset dataset --resize "!resize_value!" -d "%%c" mvtec "./data/MVTec"
        )
    )
)

echo All tests completed.