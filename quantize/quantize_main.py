'''
Author: Ayush Goel (aygoel@seas.upenn.edu)
'''
from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer,)
from neural_compressor.data import DataLoader, Datasets
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.quantization import fit
import torch

from quantize import *
from ...configuration_manager import ConfigurationManager

if __name__ == "__main__":

    config = ConfigurationManager(config_file="configurations/application.yaml")

    dataset = Datasets("tensorflow")["dummy"](shape=(1, 224, 224, 3))
    dataloader = DataLoader(framework="tensorflow", dataset=dataset)

    q_model = fit(
        model="./bert-uncased-base.pb",
        conf=PostTrainingQuantConfig(),
        calib_dataloader=dataloader,
    )

    tokenizer = BertTokenizer.from_pretrained(
    config["output_dir"], do_lower_case=config["do_lower_case"])

    model = BertForSequenceClassification.from_pretrained(config["model_dir"])
    model.to(config["device"])

    new_qconfig = torch.ao.quantization.QConfig(activation=torch.ao.quantization.default_dynamic_qconfig.activation)

    # Set the embedding layer's qconfig to the new_qconfig
    model.bert.embeddings.qconfig = new_qconfig

    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    print(f"Size of original model: {ModelSizeCalculator.calculate_and_print_model_size(model)}")
    print(f"Size of quantized model: {ModelSizeCalculator.calculate_and_print_model_size(quantized_model)}")

    ModelEvaluatorTime.evaluate_model(model, config, tokenizer)
    ModelEvaluatorTime.evaluate_model(quantized_model, config, tokenizer)
