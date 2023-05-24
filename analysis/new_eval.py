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


    with open("default_inv.jsonl", "r+") as f:
        para = f.readlines()
        para = json.loads(para[0])

        RUN_prefix = para["RUN.prefix"].split("/")

        # if "/".join(RUN_prefix[0:3]) != Config.bucket[:-1]:
        #     RUN_prefix.insert(0, Config.bucket[:-1])

        if RUN_prefix[-2] != Config.dataset:
            RUN_prefix[-2] = Config.dataset
        para["RUN.prefix"] = "/".join(RUN_prefix)

        RUN_job_name = para["RUN.job_name"].split("/")
        if RUN_job_name[-2] != Config.dataset:
            RUN_job_name[-2] = Config.dataset
        para["RUN.job_name"] = "/".join(RUN_job_name)
        f.close()

    with open("default_inv.jsonl", "w") as f:
        f.write(json.dumps(para))
        f.close()


    sweep = Sweep(RUN, Config).load("default_inv.jsonl")

    for kwargs in sweep:
        logger.print(RUN.prefix, color='green')
        jaynes.config("local")
        thunk = instr(evaluate, **kwargs)
        jaynes.run(thunk)

    jaynes.listen()