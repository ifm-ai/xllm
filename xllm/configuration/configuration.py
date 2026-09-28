from typing import Optional, Any, Set, Dict, Type, TypeVar, get_origin
from argparse import ArgumentParser
from logging import getLogger
from dataclasses import (
    field,
    fields,
    asdict,
    dataclass,
    is_dataclass,
)
import abc
import copy
import json
import argparse
import typeguard
import dataclasses


FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}

SEPARATOR = "."


logger = getLogger()


class DeepCopyDict(dict):
    def __getitem__(self, item):
        res = super(DeepCopyDict, self).__getitem__(item)
        if callable(res):
            return res()
        return copy.deepcopy(res)

    def __setitem__(self, key, value):
        if key in self:
            raise ValueError(f"{key} already in ConfStore")

        super(DeepCopyDict, self).__setitem__(key, value)


ConfStore = DeepCopyDict()


MISSING: Any = "???"


class NOTSET:
    pass


class MissingArg(Exception):
    pass


class WrongConfName(Exception):
    pass


class WrongConfType(Exception):
    pass


class WrongArgType(Exception):
    pass


class WrongFieldType(Exception):
    pass


class OptionalDataClass(Exception):
    pass


class DefaultDataClassValue(Exception):
    pass


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s == NOTSET:
        return MISSING
    elif s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def flatten_dict(to_flatten, prefix=""):
    flattened = {}
    for x, y in to_flatten.items():
        if isinstance(y, dict):
            flattened.update(flatten_dict(y, prefix=f"{prefix}{x}{SEPARATOR}"))
        else:
            flattened[f"{prefix}{x}"] = y
    return flattened


def is_optional(some_type):
    return some_type == Optional[some_type]


def get_optional_type(some_type):
    if is_optional(some_type):
        return some_type.__args__[0]
    return some_type


def NOCLI(default: Any = None):
    return field(default=default, metadata={"NOCLI": True})


def is_nocli(some_field):
    return (
        isinstance(some_field, dataclasses.Field)
        and some_field.metadata.get("NOCLI", False)
    )


def _get_default_value(field_: dataclasses.Field, default_instance: Optional["Config"]):
    default_value = (
        (
            field_.default_factory()
            if field_.default_factory != dataclasses.MISSING
            else field_.default
        )
        if default_instance is None
        else getattr(default_instance, field_.name)
    )
    if default_value == dataclasses.MISSING:
        return MISSING
    return default_value


@dataclass
class Config(abc.ABC):
    def __new__(cls, *args, **kwargs):
        for field in fields(cls):
            ftype = get_optional_type(field.type)
            if get_origin(ftype) is dict:
                raise RuntimeError(
                    f"Field {field.name} of class {cls.__name__} is of type {ftype}, which breaks our Config."
                )
            if is_dataclass(ftype) and isinstance(field.default, Config):
                raise DefaultDataClassValue(
                    f"Field {field.name} of class {cls.__name__} is set to a shared "
                    f"default config. Setting a shared default value is dangerous "
                    f"and can lead to unexpected changes across configs. "
                    f"Use `default_factory` instead."
                )
            if field.default_factory != dataclasses.MISSING:
                try:
                    value = field.default_factory()
                    typeguard.check_type(value, field.type)
                except TypeError:
                    raise DefaultDataClassValue(
                        f"`default_factory` for field {field.name} of class {cls.__name__} "
                        f"should build objects of type {field.type.__name__}."
                    )
        return super().__new__(cls)

    @classmethod
    def to_cli(cls, prefix: str = "", parser: Optional[ArgumentParser] = None) -> ArgumentParser:
        if parser is None:
            parser = ArgumentParser(allow_abbrev=False)
            parser.add_argument("--cfg", type=str)

        for field in fields(cls):
            if prefix == "":
                assert field.name != "cfg", "'cfg' field is reserved for cli parser"
            fullname = f"{prefix}{field.name}"
            field_type = get_optional_type(field.type)

            if is_nocli(field):
                pass
            elif is_dataclass(field_type):
                field_type.to_cli(prefix=f"{fullname}.", parser=parser)
                parser.add_argument(f"--{fullname}", type=str, default=NOTSET)
            else:
                parser.add_argument(
                    f"--{fullname}",
                    type=bool_flag if field_type == bool else field_type,
                    help="" if field.metadata is None else field.metadata.get("help"),
                    default=NOTSET,
                )

        return parser

    @classmethod
    def from_cli(
        cls,
        arg_dict: Dict[str, Any],
        prefix: str = "",
        default_instance: Optional["Config"] = None,
        allow_incomplete: Optional[bool] = False,
    ):
        assert default_instance is None or isinstance(default_instance, cls)

        kwargs = {}
        used_args: Set[str] = {"cfg"}
        for field in fields(cls):
            fullname = f"{prefix}{field.name}"
            field_type = get_optional_type(field.type)
            assert allow_incomplete or (fullname not in arg_dict) == is_nocli(field)
            cli_value = NOTSET
            if fullname in arg_dict:
                cli_value = arg_dict[fullname]
                used_args |= {k for k in arg_dict if k.startswith(fullname)}

            # default value is field not set in the CLI
            default_value = _get_default_value(field, default_instance)

            if is_nocli(field):
                kwargs[field.name] = default_value
            elif is_dataclass(field_type):
                must_recurse = any(
                    [
                        x.startswith(fullname) and y != NOTSET
                        for x, y in arg_dict.items()
                    ]
                )
                if not must_recurse and is_optional(field.type):
                    kwargs[field.name] = default_value
                else:
                    if cli_value == NOTSET:
                        sub_conf = None if default_value == MISSING else default_value
                    elif isinstance(cli_value, str) and cli_value != MISSING:
                        if cli_value not in ConfStore:
                            raise WrongConfName(
                                f"Unknown conf key {cli_value} for field {fullname}"
                            )
                        sub_conf = ConfStore[cli_value]
                    else:
                        raise WrongArgType(f"Value for {fullname} should be a string!")

                    # check type
                    if sub_conf is not None and not isinstance(sub_conf, field_type):
                        raise WrongConfType(
                            f"Invalid configuration. Provided a configuration of type "
                            f'"{type(sub_conf).__name__}", expected "{field_type.__name__}".'
                        )

                    # recursively set field value
                    kwargs[field.name] = field_type.from_cli(
                        arg_dict=arg_dict,
                        prefix=f"{fullname}.",
                        default_instance=sub_conf,
                    )
            elif cli_value != NOTSET:
                try:
                    kwargs[field.name] = field_type(cli_value)
                except ValueError as e:
                    raise WrongArgType(e)
            else:
                kwargs[field.name] = default_value

        # check unused args
        cls.check_unused_args(arg_dict, used_args, prefix, False)
        # check missing values
        cls.check_missing(kwargs, prefix)

        return cls(**kwargs)

    @classmethod
    def from_flat(
        cls,
        flat: Dict[str, Any],
        prefix: str = "",
        default_instance: Optional["Config"] = None,
    ):
        kwargs: Dict[str, Any] = {}
        used_args: Set[str] = set()
        for field in fields(cls):
            fullname = f"{prefix}{field.name}"
            field_type = get_optional_type(field.type)

            default_value = _get_default_value(field, default_instance)

            if is_dataclass(field_type):
                sub_conf = None if default_value == MISSING else default_value

                if is_optional(field.type) and fullname in flat:
                    assert flat[fullname] is None
                    assert (
                            len({k for k in flat if k.startswith(f"{fullname}{SEPARATOR}")})
                            == 0
                    )
                    kwargs[field.name] = None
                else:
                    kwargs[field.name] = field_type.from_flat(
                        flat, prefix=f"{fullname}{SEPARATOR}", default_instance=sub_conf
                    )
                    used_args |= {
                        k for k in flat if k.startswith(f"{fullname}{SEPARATOR}")
                    }
            else:
                try:
                    kwargs[field.name] = flat[fullname]
                    used_args.add(fullname)
                except KeyError:
                    if default_value == MISSING:
                        raise MissingArg(
                            f"Arg {fullname} unspecified and has no default"
                        )
                    else:
                        kwargs[field.name] = default_value

        # check unused args
        cls.check_unused_args(flat, used_args, prefix, True)
        # check missing values
        cls.check_missing(kwargs, prefix)

        return cls(**kwargs)

    @classmethod
    def from_dict(cls: Type["Config"], src: Dict):
        flat_dict = flatten_dict(src)
        return cls.from_flat(flat_dict)

    @classmethod
    def from_json(cls: Type["Config"], s: str):
        return cls.from_dict(json.loads(s))

    @classmethod
    def check_unused_args(cls, arg_dict: Dict[str, Any], used_args: Set[str], prefix: str, warning_only: bool):
        unused_args = {
            k: v for k, v in arg_dict.items() if (k not in used_args) and k.startswith(prefix)
        }
        if unused_args:
            if warning_only:
                logger.warning(
                    f"Some fields are unused to instantiate {cls}: {unused_args}"
                )
            else:
                raise RuntimeError(
                    f"Some fields are unused to instantiate {cls}: {unused_args}"
                )

    @classmethod
    def check_missing(cls, arg_dict: Dict[str, Any], prefix: str):
        for x, y in arg_dict.items():
            if y == MISSING:
                raise MissingArg(f"Arg {prefix}{x} is MISSING")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_flat(self):
        return flatten_dict(asdict(self))

    def to_json(self: "Config"):
        return json.dumps(asdict(self), sort_keys=True, indent=4)


T = TypeVar("T", bound=Config)


def cfg_from_cli(
    base_config: Optional[T] = None,
    schema: Optional[Type[T]] = None
) -> T:
    assert (base_config is None) == (schema is not None)
    if base_config is not None:
        schema = base_config.__class__

    assert schema is not None
    arg_dict = vars(schema.to_cli().parse_args())
    if arg_dict["cfg"] is not None:
        base_config = ConfStore[arg_dict.pop("cfg")]
        assert isinstance(base_config, schema)
    cfg = schema.from_cli(arg_dict=arg_dict, default_instance=base_config)
    assert isinstance(cfg, schema)
    return cfg
