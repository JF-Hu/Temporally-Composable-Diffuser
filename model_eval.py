import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == '__main__':
    from ml_logger import logger, instr, needs_relaunch
    from analysis import RUN
    import jaynes
    from scripts.evaluate_inv_parallel import evaluate
    from config.locomotion_config import Config
    from params_proto.neo_hyper import Sweep


    sweep = Sweep(RUN, Config).load("analysis/default_inv_copy.jsonl")

    for kwargs in sweep:
        logger.print(RUN.prefix, color='green')
        jaynes.config("local")
        thunk = instr(evaluate, **kwargs)
        jaynes.run(thunk)

    # jaynes.listen()
