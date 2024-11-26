import distutils.util
import json
import os.path
from types import SimpleNamespace as Namespace


def getattr_recursive(obj, s):
    if isinstance(s, list):
        split = s
    else:
        split = s.split('/')

    try:
        return getattr_recursive_(obj, split)
    except KeyError:
        split.insert(0, 'params')
        return getattr_recursive_(obj, split)


def getattr_recursive_(obj, split):
    if isinstance(obj, dict):
        if len(split) > 1:
            return getattr_recursive(obj[split[0]], split[1:])
        else:
            return obj[split[0]]
    return getattr_recursive(getattr(obj, split[0]), split[1:]) if len(split) > 1 else getattr(obj, split[0])


def setattr_recursive(obj, s, val):
    if not isinstance(s, list):
        s = s.split('/')

    if isinstance(obj, dict):
        if not s[0] in obj:
            s.insert(0, 'params')
        if len(s) > 1:
            return setattr_recursive(obj[s[0]], s[1:], val)
        else:
            obj[s[0]] = val
            return None
    if not hasattr(obj, s[0]):
        s.insert(0, 'params')
    return setattr_recursive(getattr(obj, s[0]), s[1:], val) if len(s) > 1 else setattr(obj, s[0],
                                                                                        val)


def get_bool_user(message, default: bool):
    resp = input(f'{message} {"[Y/n]" if default else "[y/N]"}\n')
    try:
        return distutils.util.strtobool(resp)
    except ValueError:
        return default


def override_params(params, overrides):
    for override in overrides:
        try:
            oldval = getattr_recursive(params, override[0])
            if type(oldval) == bool:
                to_val = bool(distutils.util.strtobool(override[1]))
            else:
                to_val = type(oldval)(override[1])
            setattr_recursive(params, override[0],
                              to_val)
            print("Overriding param", override[0], "from", oldval, "to", to_val)
        except (KeyError, AttributeError):
            print("Could not override", override[0], "as it does not exist. Aborting.")
            exit(1)

    return params


def load_config(config_path, overrides=None):
    with open(config_path, 'r') as f:
        params = json.load(f, object_hook=lambda d: Namespace(**d))

    if overrides is not None:
        params = override_params(params, overrides)

    return params


def save_params(params, config_path, force=False):
    if not force and os.path.exists(config_path):
        resp = get_bool_user(f"File exists at {config_path}. Override?", False)
        if not resp:
            print("Chose not to override.")
            return
    # js = params.model_dump()

    with open(config_path, 'w') as f:
        json_data = params.to_json(indent=4)
        f.write(json_data)
        f.close()

    print(f"Saved params to {config_path}.")