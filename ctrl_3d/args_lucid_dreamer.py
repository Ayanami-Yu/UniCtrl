from ctrl_3d.LucidDreamer.arguments import ParamGroup


class CtrlParams(ParamGroup):
    def __init__(self, parser):
        self.use_plain_cfg = False
        self.guidance_type = "static"
        self.w_src = 1.0
        self.w_tgt = 1.0
        self.w_src_ctrl_type = "static"
        self.w_tgt_ctrl_type = "static"
        self.t_ctrl_start = None

        self.src_prompt = ""
        self.tgt_prompt = ""

        self.ctrl_mode = "add"
        self.removal_version: int = 1

        super().__init__(parser, "PromptCtrl Parameters")
