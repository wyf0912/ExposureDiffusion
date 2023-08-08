python3 test_SID.py --model eld_iter_model --model_path checkpoints/sid_PGru_naf2/model_300_00386400.pt --concat_origin --adaptive_res_and_x0 --with_photon -r --include 4 --netG naf2

python3 test_SID.py --model eld_iter_model --model_path checkpoints/sid_Pg_naf2/model_300_00386400.pt --concat_origin --adaptive_res_and_x0 --with_photon -r --include 4 --netG naf2

python3 test_SID.py --model eld_iter_model --model_path checkpoints/sid_Pg/model_300_00386400.pt --concat_origin --adaptive_res_and_x0 --with_photon -r --include 4 --iter_num 2


python3 test_SID.py --model eld_iter_model --model_path checkpoints/sid_real/model_300_00386400.pt --concat_origin --adaptive_res_and_x0 --with_photon -r --include 4