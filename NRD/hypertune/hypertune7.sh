python glove_train_multispace.py --DX1_dim 50 --DX_dim 100 --PR_dim 50 --penalty 0.5 --penalty_metric cosine --count_cap 20 --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --save_folder elder/embed_mats/ --job_index 7
python glove_train_multispace.py --DX1_dim 50 --DX_dim 100 --PR_dim 100 --penalty 0.0 --penalty_metric cosine --count_cap 20 --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --save_folder elder/embed_mats/ --job_index 7
python glove_train_multispace.py --DX1_dim 50 --DX_dim 100 --PR_dim 100 --penalty 1.0 --penalty_metric cosine --count_cap 20 --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --save_folder elder/embed_mats/ --job_index 7
python glove_train_multispace.py --DX1_dim 50 --DX_dim 200 --PR_dim 50 --penalty 0.5 --penalty_metric cosine --count_cap 20 --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --save_folder elder/embed_mats/ --job_index 7
python glove_train_multispace.py --DX1_dim 50 --DX_dim 200 --PR_dim 100 --penalty 0.0 --penalty_metric cosine --count_cap 20 --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --save_folder elder/embed_mats/ --job_index 7
python glove_train_multispace.py --DX1_dim 50 --DX_dim 200 --PR_dim 100 --penalty 1.0 --penalty_metric cosine --count_cap 20 --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --save_folder elder/embed_mats/ --job_index 7
python glove_train_multispace.py --DX1_dim 100 --DX_dim 100 --PR_dim 50 --penalty 0.5 --penalty_metric cosine --count_cap 20 --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --save_folder elder/embed_mats/ --job_index 7
python glove_train_multispace.py --DX1_dim 100 --DX_dim 100 --PR_dim 100 --penalty 0.0 --penalty_metric cosine --count_cap 20 --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --save_folder elder/embed_mats/ --job_index 7
python glove_train_multispace.py --DX1_dim 100 --DX_dim 100 --PR_dim 100 --penalty 1.0 --penalty_metric cosine --count_cap 20 --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --save_folder elder/embed_mats/ --job_index 7
python glove_train_multispace.py --DX1_dim 100 --DX_dim 200 --PR_dim 50 --penalty 0.5 --penalty_metric cosine --count_cap 20 --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --save_folder elder/embed_mats/ --job_index 7
python glove_train_multispace.py --DX1_dim 100 --DX_dim 200 --PR_dim 100 --penalty 0.0 --penalty_metric cosine --count_cap 20 --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --save_folder elder/embed_mats/ --job_index 7
python glove_train_multispace.py --DX1_dim 100 --DX_dim 200 --PR_dim 100 --penalty 1.0 --penalty_metric cosine --count_cap 20 --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --save_folder elder/embed_mats/ --job_index 7
python glove_train_multispace.py --DX1_dim 200 --DX_dim 100 --PR_dim 50 --penalty 0.5 --penalty_metric cosine --count_cap 20 --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --save_folder elder/embed_mats/ --job_index 7
python glove_train_multispace.py --DX1_dim 200 --DX_dim 100 --PR_dim 100 --penalty 0.0 --penalty_metric cosine --count_cap 20 --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --save_folder elder/embed_mats/ --job_index 7
python glove_train_multispace.py --DX1_dim 200 --DX_dim 100 --PR_dim 100 --penalty 1.0 --penalty_metric cosine --count_cap 20 --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --save_folder elder/embed_mats/ --job_index 7
python glove_train_multispace.py --DX1_dim 200 --DX_dim 200 --PR_dim 50 --penalty 0.5 --penalty_metric cosine --count_cap 20 --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --save_folder elder/embed_mats/ --job_index 7
python glove_train_multispace.py --DX1_dim 200 --DX_dim 200 --PR_dim 100 --penalty 0.0 --penalty_metric cosine --count_cap 20 --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --save_folder elder/embed_mats/ --job_index 7
python glove_train_multispace.py --DX1_dim 200 --DX_dim 200 --PR_dim 100 --penalty 1.0 --penalty_metric cosine --count_cap 20 --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --save_folder elder/embed_mats/ --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 100 --PR_dim 50 --fc_width 256 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 100 --PR_dim 50 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 100 --PR_dim 50 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 100 --PR_dim 50 --fc_width 512 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 100 --PR_dim 50 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 100 --PR_dim 50 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 100 --PR_dim 100 --fc_width 256 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 100 --PR_dim 100 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 100 --PR_dim 100 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 100 --PR_dim 100 --fc_width 512 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 100 --PR_dim 100 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 100 --PR_dim 100 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 200 --PR_dim 50 --fc_width 256 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 200 --PR_dim 50 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 200 --PR_dim 50 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 200 --PR_dim 50 --fc_width 512 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 200 --PR_dim 50 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 200 --PR_dim 50 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 200 --PR_dim 100 --fc_width 256 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 200 --PR_dim 100 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 200 --PR_dim 100 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 200 --PR_dim 100 --fc_width 512 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 200 --PR_dim 100 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 50 --DX_dim 200 --PR_dim 100 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 100 --PR_dim 50 --fc_width 256 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 100 --PR_dim 50 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 100 --PR_dim 50 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 100 --PR_dim 50 --fc_width 512 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 100 --PR_dim 50 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 100 --PR_dim 50 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 100 --PR_dim 100 --fc_width 256 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 100 --PR_dim 100 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 100 --PR_dim 100 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 100 --PR_dim 100 --fc_width 512 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 100 --PR_dim 100 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 100 --PR_dim 100 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 200 --PR_dim 50 --fc_width 256 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 200 --PR_dim 50 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 200 --PR_dim 50 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 200 --PR_dim 50 --fc_width 512 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 200 --PR_dim 50 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 200 --PR_dim 50 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 200 --PR_dim 100 --fc_width 256 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 200 --PR_dim 100 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 200 --PR_dim 100 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 200 --PR_dim 100 --fc_width 512 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 200 --PR_dim 100 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 100 --DX_dim 200 --PR_dim 100 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 100 --PR_dim 50 --fc_width 256 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 100 --PR_dim 50 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 100 --PR_dim 50 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 100 --PR_dim 50 --fc_width 512 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 100 --PR_dim 50 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 100 --PR_dim 50 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 100 --PR_dim 100 --fc_width 256 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 100 --PR_dim 100 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 100 --PR_dim 100 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 100 --PR_dim 100 --fc_width 512 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 100 --PR_dim 100 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 100 --PR_dim 100 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 200 --PR_dim 50 --fc_width 256 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 200 --PR_dim 50 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 200 --PR_dim 50 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 200 --PR_dim 50 --fc_width 512 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 200 --PR_dim 50 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 200 --PR_dim 50 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 200 --PR_dim 100 --fc_width 256 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 200 --PR_dim 100 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 200 --PR_dim 100 --fc_width 256 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 200 --PR_dim 100 --fc_width 512 --lr1 0.0002 --lr2 1e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.5 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 200 --PR_dim 100 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 0.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
python template_all_multispace0102.py --model_name setsum_nn --DX1_dim 200 --DX_dim 200 --PR_dim 100 --fc_width 512 --lr1 0.0002 --lr2 2e-05 --dropout 0.3 --batchsize 512 --embed_file pretrain --penalty 1.0 --penalty_metric cosine --count_cap 20 --tst_seed 0 --cohort ami --dx1_rarecutpoint 10 --dx_rarecutpoint 10 --pr_rarecutpoint 10 --other_pred 0 --ndxpr 0 --val_fold 7 --result_file output/ht_result0216_{}.csv --job_index 7
