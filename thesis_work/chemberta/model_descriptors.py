from typing import Union

import numpy as np
import pandas as pd
import torch
from simpletransformers.language_representation import RepresentationModel
from transformers import RobertaModel, RobertaTokenizer, RobertaTokenizerFast

from thesis_work.utils.initialization import check_initialization_params
from thesis_work.utils.utils import check_device


def initialize_model_tokenizer(model_name: str = "DeepChem/ChemBERTa-77M-MLM"):
    # tokenizer = RobertaTokenizerFast.from_pretrained('seyonec/PubChem10M_SMILES_BPE_450k')
    # model = RobertaModel.from_pretrained('seyonec/ChemBERTa-77M-MLM', output_hidden_states = True)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    return model, tokenizer


def get_model_descriptor(
    model: RobertaModel,
    tokenizer: Union[RobertaTokenizer, RobertaTokenizerFast],
    smiles_str: str,
) -> np.ndarray:
    """
    NOTE:
        - This is the only working method that doesn't consume all the memory.
        The important path is that, we tokenize the SMILES strings one by one within tokenizer.
        Declaring tokenizer with all inputs results in memory error.

        - Vector size: [65, 384]
            65: Token length
            384: Embedding size

    """

    # NOTE: Not working code
    # token = tokenizer(smile_strings, return_tensors='pt', padding=True, truncation=True)

    # torch.set_num_threads(1)
    token = torch.tensor(
        [
            tokenizer.encode(
                smiles_str,
                add_special_tokens=True,
                max_length=512,  # NOTE: Can be 384
                padding=True,
                truncation=True,
            )
        ]
    )
    with torch.no_grad():
        output = model(token)

        # TODO: Should we need this?
        # output = output[0][:, 0, :]  # Take the [CLS] token's embedding for each sentence

    # NOTE: Same with output[0]
    last_layer = output.last_hidden_state

    # return torch.mean(sequence_out[0], dim=0).tolist()
    return torch.mean(last_layer[0], dim=0).detach().numpy()


def get_model_descriptors(
    smiles_series: pd.Series,
    model_name: str = "DeepChem/ChemBERTa-77M-MLM",
    method: str = "simpletransformers",
    combine_strategy: str = "mean",
    device: str = "cuda",
) -> np.array:
    """Calculates and returns model vector embedding for given smiles list.

    Args:
        smiles_series: Smiles data.
        model_name: Model name to be used.
        method: Method to be used for embedding.
        combine_strategy = None -> word_embedding / "mean" -> sentence_embedding

    NOTE
        - For 68,000 compounds with method: simpletransformers it takes:
            - 19s on GPU
            - 4m 23 on CPU
        - For simpletransformers methods, returning `tolist()` is important hence, we
        embed them in dataframe.
    """

    check_device(device=device)
    use_cuda = device == "cuda"

    check_initialization_params(
        attr=method, accepted_list=["manual", "simpletransformers"]
    )

    if method == "manual":
        model, tokenizer = initialize_model_tokenizer(model_name)

        return smiles_series.apply(
            lambda x: get_model_descriptor(
                model=model, tokenizer=tokenizer, smiles_str=x
            )
        )
    elif method == "simpletransformers":
        model = RepresentationModel(
            model_type="roberta", model_name=model_name, use_cuda=use_cuda
        )
        return model.encode_sentences(smiles_series, combine_strategy=combine_strategy)


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    """
    From https://github.com/seyonechithrananda/bert-loves-chemistry/blob/master/chemberta/visualization/ChemBERTA_dimensionaliy_reduction_BBBP.ipynb
    """

    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]

    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def gen_embeddings(model, tokenizer, smiles):
    """
    From https://github.com/seyonechithrananda/bert-loves-chemistry/blob/master/chemberta/visualization/ChemBERTA_dimensionaliy_reduction_BBBP.ipynb

    NOTE: Not working since tokenizer inputs all smiles at once.
    """

    # Tokenize sentences
    encoded_input = tokenizer(
        smiles, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, mean pooling
    return mean_pooling(model_output, encoded_input["attention_mask"])
