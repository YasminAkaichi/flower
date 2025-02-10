import torch
import numpy as np
#from transformers import GPT2Tokenizer
from typing import  Callable, Dict, List, Optional, Tuple, Union
from andante.collections import OrderedSet
from flwr.common import (
    Parameters,
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    ndarray_to_bytes,
)
import numpy as np
import pickle
from io import BytesIO
from typing import List
from andante.program import AndanteProgram 
from andante.solver import AndanteSolver
from andante.parser import Parser
from andante.collections import OrderedSet

def serialize_ordered_set(ordered_set: OrderedSet) -> bytes:
    """Serialize OrderedSet to bytes."""
    return pickle.dumps(ordered_set)

def deserialize_ordered_set(serialized: bytes) -> OrderedSet:
    """Deserialize bytes back to OrderedSet."""
    return pickle.loads(serialized)

def ordered_set_to_ndarray(ordered_set: OrderedSet) -> np.ndarray:
    """Convert OrderedSet to NumPy array."""
    return np.array(list(ordered_set))

def ndarray_to_ordered_set(array: np.ndarray) -> OrderedSet:
    """Convert NumPy array back to OrderedSet."""
    return OrderedSet(array.tolist())


def text_to_tensor(rules: str|OrderedSet) -> torch.Tensor:
    """Convert a text or an OrderedSet to PyTorch tensor."""
    if isinstance(rules,str):
        tokens = [ord(char) for char in rules]
    elif isinstance(rules,OrderedSet):
        text = OrderedSet.__str__(rules)
        tokens = [ord(char) for char in text]
    else:
        raise ValueError("Input must be either a string or an OrderedDict")
    # Convert the tokens to a PyTorch tensor
    tensor = torch.tensor(tokens)
    return tensor

def tensor_to_parameters(tensor:torch.tensor) -> Parameters:
    """Convert a PyTorch tensor to Parameters."""
    # Serialize the tensor to bytes
    tensor_bytes = ndarray_to_bytes(tensor)
    # Create Parameters instance
    parameters = Parameters(tensors=[tensor_bytes], tensor_type="tensor_type")
    return parameters

#not used
def parameters_to_tensor(parameters:Parameters) -> torch.tensor:
    """Convert Parameters to PyTorch tensor."""
    # Deserialize the bytes to numpy array
    tensor_numpy = torch.from_numpy(bytearray(parameters.tensors[0])).view(-1)

    # Convert numpy array to PyTorch tensor
    tensor = tensor_numpy.type(parameters.tensor_type)
    
    return tensor

def tensor_to_text(tensor: torch.Tensor) -> str:
    """Convert a PyTorch tensor to text."""
    # Convert the tensor to a list of integers
    token_ids = tensor.tolist()   
    # Flatten the nested list structure
    #flattened_token_ids = [item for sublist in tensor.tolist() for item in sublist]
    # Flatten the list if it's nested
    if any(isinstance(sublist, list) for sublist in token_ids):
        token_ids = [item for sublist in token_ids for item in sublist]
    # Decode the token IDs into characters
    text = ''.join([chr((token_id)) for token_id in token_ids])
    return text


def get_parameters(results:OrderedSet) -> List[np.ndarray]:
    """ Convert parameters to NumPy arrays """
    if results is None:
        return None
    else:
        #convert orderest Set to text then to tensor
        tensor = text_to_tensor(results)
        #convert tensors to parameters
        parameters = tensor_to_parameters(tensor)
        return parameters_to_ndarrays(parameters)



def text_to_ordered_set(text: str) -> 'OrderedSet':
    lines = text.strip().split('\n')
    ordered_set = OrderedSet()
    for line in lines:
        line = line.strip()
        if line:
            ordered_set.add(line)
    return ordered_set

def set_parameters(ilp: AndanteProgram , parameters: List[np.ndarray]):
    """ Set the parameters for the ILP """
    if parameters and len(parameters) > 0:
        array_value = parameters[0]
    else:
        array_value = parameters
    #convert NDarray parameters to tensors
    tensor_value = torch.tensor(array_value)
    #convert tensors to text 
    text = tensor_to_text(tensor_value)
    #convert text to OrderedSet 
    ilp.results = OrderedSet.text_to_ordered_set(text)


def aggregate_fedilp(results: List[Tuple[np.ndarray, int]]) -> np.ndarray:
    """Do union to rules"""
    # step 1 convert list of ndarrays to tensors 
    tensors = [torch.tensor(result[0]) for result in results]
    # step 2 convert tensors to text 
    texts = [tensor_to_text(tensor) for tensor in tensors]
    # step 3 convert text to OrderedSet 
    ordered_sets= [OrderedSet.text_to_ordered_set(text) for text in texts]
    # step 4 do the union to rules 
    union_set = OrderedSet()
    for ordered_set in ordered_sets:
        union_set |= ordered_set
    # step5: convert orderedset to text then to tensors then to ndarray 
    text_union = OrderedSet.__str__(union_set)
    #step6: convert text back to tensors
    tensor_union = text_to_tensor(text_union)
    #step7: convert tensors back to ndarrays ( maybe to ndarray directly ? but before to parameters) 
    tensor_parameters = tensor_to_parameters(tensor_union)
    ndarrays = parameters_to_ndarrays(tensor_parameters) 
    return ndarrays
