from configparser import ConfigParser
import logging 

class ConfigFileparser:
    """
    Parses configfile and stores them in attributes
    """
    def __init__(self, configfile = "configfile.ini"):
        parser = ConfigParser()
        parser.read(configfile)
        self.class_count = int(parser.get('DEFAULT','CLASS_COUNT',fallback=39))
        self.gan_model = parser.get('DEFAULT','GAN_MODEL',fallback="STYLEGAN")
        self.classifier_network = parser.get('DEFAULT','CLASSIFIER_MODEL',fallback="VGG")
        self.shift_predictor_model = parser.get('DEFAULT','SHIFT_PREDICTOR_MODEL',fallback="2")
        self.subset_labels_str = parser.get('DEFAULT','SUBSET_LABELS',fallback="3, 19, 30, 38")
        self.subset_labels = [int(i) for i in self.subset_labels_str.split(",")]

        self.faithfulness_ratio = float(parser.get('HYPERPARAMETERS','HYPERPARAMETERS',fallback=0.05))
        self.batch_size = int(parser.get('HYPERPARAMETERS','BATCH_SIZE',fallback=4))
        self.n_steps = int(parser.get('HYPERPARAMETERS','N_STEPS',fallback=100000))
        self.shift_predictor_lr = float(parser.get('HYPERPARAMETERS','SHIFT_PREDICTOR_LR',fallback=0.00001))
        self.noise_scale = float(parser.get('HYPERPARAMETERS','NOISE_SCALE',fallback=0.1))

        self.gan_weight_file = parser.get(self.gan_model,'GAN_WEIGHT_FILE',fallback='GAN_models/trained_weights/stylegan2-ffhq-config-f.pt')
        self.latent_size = int(parser.get(self.gan_model,'LATENT_SIZE',fallback=512))
        self.gan_resolution = int(parser.get(self.gan_model,'GAN_RESOLUTION',fallback=1024))
        self.gan_output_channel = int(parser.get(self.gan_model,'GAN_OUTPUT_CHANNEL',fallback=3))

        self.classifier_weight_file = parser.get(self.classifier_network,'CLASSIFIER_WEIGHT_FILE',fallback='classifier_weights/celebA_MultiLabels_vgg11_classifier.pt')
        self.classifier_input_size = int(parser.get(self.classifier_network,'CLASSIFIER_INPUT_SIZE',fallback=256))
        self.classifier_input_channel = int(parser.get(self.classifier_network,'CLASSIFIER_INPUT_CHANNEL',fallback=3))

        self.shift_predictor_folder = parser.get('SHIFT_PREDICTOR','SHIFT_PREDICTOR_WEIGHT_FILE',fallback='trained_weights/shift_in_w_Bald_Male_Multi_3_19_0.050.pt')
        self.shift_predictor_weight_file = self.shift_predictor_folder + self.gan_model + "_" + self.classifier_network + "_" + self.shift_predictor_model + ".pt"
        self.load_pretrained = parser.getboolean('SHIFT_PREDICTOR','LOAD_PRETRAINED',fallback=True)

        log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        self.logger = logging.getLogger('CounterFactuals')
        fileHandler = logging.FileHandler("debug.log")
        fileHandler.setFormatter(log_formatter)
        self.logger .addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(log_formatter)
        self.logger .addHandler(consoleHandler)

        self.logger.setLevel(logging.DEBUG)
        self.logger.debug("%s", self.__dict__)