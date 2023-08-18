"""Function and class initializations related utilities."""
import inspect
from typing import Any, Callable, Dict, List, TypeVar

T1 = TypeVar("T1")


def check_initialization_params(attr: T1, accepted_list: List[T1]) -> None:
    """Check whether input attribute in accepted_list or not.

    Args:
        attr: Tested attribute
        accepted_list: List of accepted attribute values

    Returns:
        Nothing for success. Otherwise raises error.

    Raises:
        ValueError: If input is not in the accepted list.
    """
    if attr not in accepted_list:
        raise ValueError(f"{attr} should be within {accepted_list}")


def get_function_params(function: Callable) -> Any:
    """Get function parameters.

    Args:
        function: Function to be checked for its parameters.

    Returns:
        Given function parameters.
    """
    return inspect.signature(function).parameters.values()


def get_function_param_names(function: Callable) -> List[str]:
    """Get parameter names of the given function.

    Args:
        function: Function to be used.

    Returns:
        Function parameters names.
    """
    params = get_function_params(function)

    return [param.name for param in params]


def get_function_defaults(function: Callable) -> Dict:
    """Get default values of the given function.

    Args:
        function: Function to be used.

    Returns:
        Function parameter default dictionary.
    """
    params = get_function_params(function)

    return {x.name: x.default for x in params if x.default != inspect._empty}


def check_function_init_params(function: Callable, init_params: Dict) -> None:
    """Check whether input init_params in accepted_params or not.

    Args:
        function: Function to be used.
        init_params: Parameters to be checked.

    Returns:
        Nothing for success. Otherwise raises error.

    Raises:
        ValueError: If input is not in the accepted list.
    """
    if init_params == {}:
        return

    function_params = get_function_param_names(function)

    for key in init_params.keys():
        check_initialization_params(attr=key, accepted_list=function_params)
