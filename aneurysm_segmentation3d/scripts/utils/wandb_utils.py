import shutil
import os
import subprocess


class WandbUrls:
    def __init__(self, url):

        hash = url.split("/")[-2]
        project = url.split("/")[-3]
        entity = url.split("/")[-4]

        self.weight_url = url
        self.log_url = (
            "https://app.wandb.ai/{}/{}/runs/{}/logs".format(
                entity, project, hash
            )
        )
        self.chart_url = "https://app.wandb.ai/{}/{}/runs/{}".format(
            entity, project, hash
        )
        self.overview_url = (
            "https://app.wandb.ai/{}/{}/runs/{}/overview".format(
                entity, project, hash
            )
        )
        self.hydra_config_url = "https://app.wandb.ai/{}/{}/runs/{}/files/hydra-config.yaml".format(
            entity, project, hash
        )
        self.overrides_url = "https://app.wandb.ai/{}/{}/runs/{}/files/overrides.yaml".format(
            entity, project, hash
        )

    def __repr__(self):
        msg = "=================================================== WANDB URLS ===================================================================\n"
        for k, v in self.__dict__.items():
            msg += "{}: {}\n".format(k.upper(), v)
        msg += "=================================================================================================================================\n"
        return msg


class Wandb:
    IS_ACTIVE = False

    @staticmethod
    def set_urls_to_model(model, url):
        wandb_urls = WandbUrls(url)
        model.wandb = wandb_urls

    @staticmethod
    def _set_to_wandb_args(wandb_args, cfg, name):
        var = getattr(cfg.wandb, name, None)
        if var:
            wandb_args[name] = var

    @staticmethod
    def launch(cfg, launch: bool):
        if launch:
            import wandb

            Wandb.IS_ACTIVE = True

            model_config = getattr(cfg.models, cfg.model_name, None)
            model_class = getattr(model_config, "class")
            tested_dataset_class = getattr(cfg.data, "class")
            otimizer_class = getattr(
                cfg.training.optim.optimizer, "class"
            )
            # scheduler_class = getattr(cfg.lr_scheduler, "class")
            features_to_include: {"mean_curvature":1, "gauss_curvature":0, "fpfh":0,"shot":0, "rf":0, "ones":0}
            mean_feat = cfg.features_to_include["mean_curvature"]
            gauss_feat = cfg.features_to_include["gauss_curvature"]
            fpfh_feat = cfg.features_to_include["fpfh"]
            shot_feat = cfg.features_to_include["shot"]
            ones_feat = cfg.features_to_include["ones"]
            tags = [
                cfg.model_name,
                # model_class.split(".")[0],
                # tested_dataset_class,
                otimizer_class,
                # scheduler_class,
                f"lr={cfg.training.optim.base_lr}",
                f"Classes={cfg.parts_to_segment}",
                cfg.wandb.notes,
                f"Mean_curv={mean_feat}",
                f"Gauss_curv={gauss_feat}",
                f"FPFH={fpfh_feat}",
                f"SHOT={shot_feat}",
                f"Ones={ones_feat}",
            ]

            wandb_args = {}
            wandb_args["project"] = cfg.wandb.project
            wandb_args["tags"] = tags
            wandb_args["resume"] = "allow"
            Wandb._set_to_wandb_args(wandb_args, cfg, "name")
            Wandb._set_to_wandb_args(wandb_args, cfg, "entity")
            Wandb._set_to_wandb_args(wandb_args, cfg, "notes")
            Wandb._set_to_wandb_args(wandb_args, cfg, "config")
            Wandb._set_to_wandb_args(wandb_args, cfg, "id")

            try:
                commit_sha = (
                    subprocess.check_output(
                        ["git", "rev-parse", "HEAD"]
                    )
                    .decode("ascii")
                    .strip()
                )
                gitdiff = subprocess.check_output(
                    ["git", "diff", "--", "':!notebooks'"]
                ).decode()
            except:
                commit_sha = "n/a"
                gitdiff = ""

            config = wandb_args.get("config", {})
            wandb_args["config"] = {
                **config,
                "run_path": os.getcwd(),
                "commit": commit_sha,
            }

            wandb.init(**wandb_args)
            # shutil.copyfile(
            #     os.path.join(os.getcwd(), ".hydra/config.yaml"),
            #     os.path.join(os.getcwd(), ".hydra/hydra-config.yaml"),
            # )
            # wandb.save(
            #     os.path.join(os.getcwd(), ".hydra/hydra-config.yaml")
            # )
            # wandb.save(
            #     os.path.join(os.getcwd(), ".hydra/overrides.yaml")
            # )

            with open("change.patch", "w") as f:
                f.write(gitdiff)
            wandb.save(os.path.join(os.getcwd(), "change.patch"))

    @staticmethod
    def add_file(file_path: str):
        if not Wandb.IS_ACTIVE:
            raise RuntimeError(
                "wandb is inactive, please launch first."
            )
        import wandb

        filename = os.path.basename(file_path)
        shutil.copyfile(
            file_path, os.path.join(wandb.run.dir, filename)
        )
