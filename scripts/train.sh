CUDA_VISIBLE_DEVICES=1 python3 train_real.py --name sid_real --model eld_iter_model --with_photon --adaptive_res_and_x0 --iter_num 2 --epoch 300 --auxloss --concat_origin --adaptive_loss


CUDA_VISIBLE_DEVICES=0 python3 train_syn.py --name sid_Pg --include 4 --noise P+g --model eld_iter_model --with_photon --adaptive_res_and_x0 --iter_num 2 --epoch 300 --auxloss --concat_origin --continuous_noise --adaptive_loss

CUDA_VISIBLE_DEVICES=1 python3 train_syn.py --name sid_Pg_e500 --include 4 --noise P+g --model eld_iter_model --with_photon --adaptive_res_and_x0 --iter_num 2 --epoch 500 --auxloss --concat_origin --continuous_noise --adaptive_loss

CUDA_VISIBLE_DEVICES=1 python3 train_syn.py --name sid_Pg_no --include 4 --noise P+g --model eld_iter_model --with_photon --adaptive_res_and_x0 --iter_num 2 --epoch 300 --auxloss --continuous_noise --adaptive_loss


CUDA_VISIBLE_DEVICES=0 python3 train_syn.py --name sid_PGru --include 4 --noise P+G+r+u --model eld_iter_model --with_photon --adaptive_res_and_x0 --iter_num 2 --epoch 300 --auxloss --concat_origin --continuous_noise --adaptive_loss

CUDA_VISIBLE_DEVICES=0 python3 train_syn.py --name sid_PGru_naf2 --include 4 --noise P+G+r+u --model eld_iter_model --with_photon --adaptive_res_and_x0 --iter_num 2 --epoch 300 --auxloss --concat_origin --continuous_noise --adaptive_loss --netG naf2

CUDA_VISIBLE_DEVICES=0 python3 train_syn.py --name sid_Pg_naf2 --include 4 --noise P+g --model eld_iter_model --with_photon --adaptive_res_and_x0 --iter_num 2 --epoch 300 --auxloss --concat_origin --continuous_noise --adaptive_loss --netG naf2

CUDA_VISIBLE_DEVICES=0 python3 train_syn.py --name sid_Pg_naf2_no --include 4 --noise P+g --model eld_iter_model --with_photon --adaptive_res_and_x0 --iter_num 2 --epoch 300 --auxloss --continuous_noise --adaptive_loss --netG naf2