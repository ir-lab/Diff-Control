from yacs.config import CfgNode as CN

cfg = CN()

cfg.mode = CN()
cfg.mode.mode = "train"
cfg.mode.model_zoo = None
cfg.mode.base_model_path = None

# Multi-gpu training
cfg.mode.num_threads = 1  # number of threads to use for data loading
cfg.mode.multiprocessing_distributed = True
cfg.mode.dist_url = "tcp://127.0.0.1:2345"  # url used to set up distributed training
cfg.mode.world_size = 1  # number of nodes for distributed training
cfg.mode.rank = 0  # node rank for distributed training
cfg.mode.dist_backend = "nccl"  # distributed backend
cfg.mode.gpu = None  # GPU id to use.
cfg.mode.do_online_eval = True  # perform online eval in every eval_freq steps
cfg.train = CN()
cfg.train.seed = 0

# Dataset
cfg.mode.parameter_path = None
cfg.train.dataset = None  # dataset to train on
cfg.train.data_path = None

# Log and save
cfg.train.log_directory = (
    "./experiments/"  # directory to save checkpoints and summaries
)
cfg.train.checkpoint_path = ""  # path to a checkpoint to load
cfg.train.log_freq = 10000  # Logging frequency in global steps
cfg.train.save_freq = 10000  # Checkpoint saving frequency in global steps

# Training
cfg.train.dim_x = None
cfg.train.dim_z = None
cfg.train.dim_a = None
cfg.train.dim_gt = None
cfg.train.sensor_len = None
cfg.train.channel_img_1 = None
cfg.train.channel_img_2 = None
cfg.train.input_size_1 = None
cfg.train.input_size_2 = None
cfg.train.input_size_3 = None
cfg.train.num_ensemble = None
cfg.train.win_size = 32  # window size
cfg.train.model_name = ""
cfg.train.batch_size = 2  # batch size
cfg.train.num_epochs = 50  # number of epochs
cfg.train.learning_rate = 1e-3  # initial learning rate
cfg.train.end_learning_rate = -1.0  # end learning rate
cfg.train.weight_decay = 1e-2  # weight decay factor for optimization
cfg.train.adam_eps = 1e-3  # epsilon in Adam optimizer
cfg.train.retrain = (
    False  # if used with checkpoint_path, will restart training from step zero
)
cfg.train.variance_focus = 0.85  # lambda in paper: [0, 1], higher value more focus on minimizing variance of error
cfg.train.loss = "mse"
cfg.train.loss_weights = [0.5, 0.5]
cfg.train.multitask = False
cfg.train.segment_classes = 55
cfg.train.task_balance = None
# Preprocessing
cfg.train.random_rotate = False  # To perform random rotation for augmentation
cfg.train.use_right = False  # randomly use right images when train on KITTI

# Online eval
cfg.train.eval_freq = 10000  # Online evaluation frequency in global steps
cfg.train.eval_summary_directory = "./experiments/"  # output directory for eval summary
cfg.train.steps_per_alpha_update = 100

# Testing
cfg.test = CN()
cfg.test.dim_x = None
cfg.test.dim_z = None
cfg.test.dim_a = None
cfg.test.dim_gt = None
cfg.test.sensor_len = None
cfg.test.channel_img_1 = None
cfg.test.channel_img_2 = None
cfg.test.input_size_1 = None
cfg.test.input_size_2 = None
cfg.test.input_size_3 = None
cfg.test.num_ensemble = None
cfg.test.win_size = 32  # window size
cfg.test.model_name = ""
cfg.test.data_path = None
cfg.test.input_height = None
cfg.test.input_width = None
cfg.test.checkpoint_path_1 = ""
cfg.test.checkpoint_path_2 = ""
cfg.test.dataset = None
cfg.test.eigen_crop = False  # crops according to Eigen NIPS14
cfg.test.garg_crop = False  # crops according to Garg  ECCV16

cfg.network = CN()
cfg.network.name = "alpha-MDF"
cfg.network.encoder = ""
cfg.network.activation_function = "ELU"

cfg.optim = CN()
cfg.optim.optim = "adamw"
cfg.optim.lr_scheduler = "polynomial_decay"
