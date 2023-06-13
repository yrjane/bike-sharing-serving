from itertools import product

def get_param_set(params: dict) -> list:
    params_keys = params.keys()
    params_values = [
        params[key] if isinstance(params[key], list) else [params[key]]
        for key in params_keys
    ]
    return [
        dict(zip(params_keys, combination))
        for combination in product(*params_values)
    ]