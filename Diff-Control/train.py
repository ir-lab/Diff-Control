import argparse
import logging
import os
import yaml
from sys import argv
from config import cfg
import controlnet_engine, BCZ_engine, BCZ_LSTM_engine, prebuild_engine, tomato_engine

# mini_controlnet_engine, BCZ_LSTM_engine
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# $python train.py --config tensegrity_1.0.yaml
logging_kwargs = dict(
    level="INFO",
    format="%(asctime)s %(threadName)s %(levelname)s %(name)s - %(message)s",
    style="%",
)
logging.basicConfig(**logging_kwargs)
logger = logging.getLogger("diffusion-pilicy")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="configuration file path", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    args = parser.parse_args()
    config_file = args.config
    if config_file and os.path.exists(config_file):
        cfg.merge_from_file(config_file)
    if bool(args.batch_size):
        cfg.train.batch_size = args.batch_size
    if bool(args.num_epochs):
        cfg.train.num_epochs = args.num_epochs
    return cfg, config_file


def main():
    cfg, config_file = parse_args()
    cfg.freeze()
    ####### check all the parameter settings #######
    logger.info("{}".format(cfg))
    logger.info("check mode - {}".format(cfg.mode.mode))
    # Create directory for logs and experiment name
    if not os.path.exists(cfg.train.log_directory):
        os.mkdir(cfg.train.log_directory)
    if not os.path.exists(os.path.join(cfg.train.log_directory, cfg.train.model_name)):
        os.mkdir(os.path.join(cfg.train.log_directory, cfg.train.model_name))
        os.mkdir(
            os.path.join(cfg.train.log_directory, cfg.train.model_name, "summaries")
        )
    else:
        logger.warning(
            "This logging directory already exists: {}. Over-writing current files".format(
                os.path.join(cfg.train.log_directory, cfg.train.model_name)
            )
        )

    ####### start the training #######
    if cfg.mode.model_zoo == "controlnet":
        train_engine = controlnet_engine.Engine(args=cfg, logger=logger)
    elif cfg.mode.model_zoo == "diffusion-model":
        train_engine = prebuild_engine.Engine(args=cfg, logger=logger)
    elif cfg.mode.model_zoo == "BCZ":
        train_engine = BCZ_engine.Engine(args=cfg, logger=logger)
    elif cfg.mode.model_zoo == "BCZ_LSTM":
        train_engine = BCZ_LSTM_engine.Engine(args=cfg, logger=logger)
    elif cfg.mode.model_zoo == "tomato-model":
        train_engine = tomato_engine.Engine(args=cfg, logger=logger)

    if cfg.mode.mode == "train":
        train_engine.train()
    if cfg.mode.mode == "pretrain":
        train_engine.train()
    if cfg.mode.mode == "test":
        train_engine.test()


if __name__ == "__main__":
    main()
