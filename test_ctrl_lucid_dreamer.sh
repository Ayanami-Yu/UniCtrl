CUDA_VISIBLE_DEVICES=1 nohup python test_ctrl_lucid_dreamer.py --opt /home/hongyu/PromptCtrl/ctrl_3d/configs/horse_girl.yaml --w_src_cli=0.9 --w_tgt_cli=0.0 --workspace_cli='horse_girl_sphere/0.9_0.0' &

CUDA_VISIBLE_DEVICES=2 nohup python test_ctrl_lucid_dreamer.py --opt /home/hongyu/PromptCtrl/ctrl_3d/configs/horse_girl.yaml --w_src_cli=0.9 --w_tgt_cli=0.3 --workspace_cli='horse_girl_sphere/0.9_0.3' &

CUDA_VISIBLE_DEVICES=3 nohup python test_ctrl_lucid_dreamer.py --opt /home/hongyu/PromptCtrl/ctrl_3d/configs/horse_girl.yaml --w_src_cli=0.9 --w_tgt_cli=0.6 --workspace_cli='horse_girl_sphere/0.9_0.6' &

CUDA_VISIBLE_DEVICES=4 nohup python test_ctrl_lucid_dreamer.py --opt /home/hongyu/PromptCtrl/ctrl_3d/configs/horse_girl.yaml --w_src_cli=0.9 --w_tgt_cli=0.9 --workspace_cli='horse_girl_sphere/0.9_0.9' &

CUDA_VISIBLE_DEVICES=5 nohup python test_ctrl_lucid_dreamer.py --opt /home/hongyu/PromptCtrl/ctrl_3d/configs/horse_girl.yaml --w_src_cli=0.9 --w_tgt_cli=1.2 --workspace_cli='horse_girl_sphere/0.9_1.2' &

CUDA_VISIBLE_DEVICES=6 nohup python test_ctrl_lucid_dreamer.py --opt /home/hongyu/PromptCtrl/ctrl_3d/configs/horse_girl.yaml --w_src_cli=0.9 --w_tgt_cli=1.5 --workspace_cli='horse_girl_sphere/0.9_1.5' &

CUDA_VISIBLE_DEVICES=7 nohup python test_ctrl_lucid_dreamer.py --opt /home/hongyu/PromptCtrl/ctrl_3d/configs/horse_girl.yaml --w_src_cli=0.9 --w_tgt_cli=1.8 --workspace_cli='horse_girl_sphere/0.9_1.8' &