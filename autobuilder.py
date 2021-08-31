#!/usr/bin/env python
import inspect
from typing import List
from typing import Union
from collections import OrderedDict
import os
import copy
import re
import glob
import argparse
import logging
from importlib import import_module
import yaml
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger()


def nice_name(label):
    tmp = label.replace('_', ' ')
    tmp = tmp[0].upper() + tmp[1:]

    def callback(pat):
        return ' ' + pat.group(1).upper()

    tmp = re.sub(r' ([a-z])', callback, tmp)
    return tmp


PARAM_TRANSLATION = {
    str: [
        'type=str',
    ],
    dict: [
        # TODO
        'type=str',
    ],
    int: [
        'type=int'
    ],
    float: [
        'type=float',
    ],
    bool: [
        'is_flag=True',
    ],
    list: [
        'type=str',  # TODO
        'multiple=True',
    ],
    List[str]: [
        'type=str',  # TODO
        'multiple=True',
    ],
    List[dict]: [
        'type=str',  # TODO
        'multiple=True',
    ],
    'file': [
        'type=click.File(\'rb+\')'
    ],
    'None': [],
}

PARAM_TRANSLATION_GALAXY = {
    str: '<param name="{name}" label="{label}" argument="{name}" type="text" {help} />',
    dict: '<param name="{name}" label="{label}" argument="{name}" type="data" format="json" {help} />',
    int: '<param name="{name}" label="{label}" argument="{name}" type="integer" value="{default}" {help} />',
    float: '<param name="{name}" label="{label}" argument="{name}" type="float" value="{default}" {help} />',
    bool: '<param name="{name}" label="{label}" argument="{name}" type="boolean" truevalue="--{name}" falsevalue="" {help} />',
    'file': '<param name="{name}" label="{label}" argument="{name}" type="data" format="data" {help} />',
    None: '<error />',
    List[str]: '<repeat name="repeat_{name}" title="{name}">\n\t\t<param name="{name}" label="{label}" argument="{name}" type="text" {help} />\n\t</repeat>',
    list: '<repeat name="repeat_{name}" title="{name}">\n\t\t<param name="{name}" label="{label}" argument="{name}" type="text" {help} />\n\t</repeat>',
    List[dict]: '<repeat name="repeat_{name}" title="{name}">\n\t\t<param name="{name}" label="{label}" argument="{name}" type="text" {help} />\n\t</repeat>',
}

PARAM_TRANSLATION_GALAXY_CLI = {
    str: {
        'opt': '#if ${name}:\n  --{name} \'${name}\'\n#end if',
        'arg': '\'${name}\'',
    },
    dict: {
        'opt': '#if ${name}:\n  --{name} \'${name}\'\n#end if',
        'arg': '\'${name}\'',
    },
    int: {
        'opt': '#if ${name}:\n  --{name} \'${name}\'\n#end if',
        'arg': '\'${name}\'',
    },
    float: {
        'opt': '#if ${name}:\n  --{name} \'${name}\'\n#end if',
        'arg': '\'${name}\'',
    },
    bool: {
        'opt': '#if ${name}:\n  ${name}\n#end if',
        'arg': '--${name}',
    },
    'file': {
        'opt': '#if ${name}:\n  --{name} \'${name}\'\n#end if',
        'arg': '\'${name}\'',
    },
    None: {
        'opt': '## UNKNOWN {name}',
        'arg': '## UNKNOWN {name}',
    },
    List[str]: {
        'opt': '#for $rep in $repeat_{name}:\n  --{name} \'$rep.{name}\'\n#end for',
        'arg': '#for $rep in $repeat_{name}:\n  --{name} \'$rep.{name}\'\n#end for',
    },
    list: {
        'opt': '#for $rep in $repeat_{name}:\n  --{name} \'$rep.{name}\'\n#end for',
        'arg': '#for $rep in $repeat_{name}:\n  --{name} \'$rep.{name}\'\n#end for',
    },
    List[dict]: {
        'opt': '#for $rep in $repeat_{name}:\n  --{name} \'$rep.{name}\'\n#end for',
        'arg': '#for $rep in $repeat_{name}:\n  --{name} \'$rep.{name}\'\n#end for',
    },
}


class ScriptBuilder(object):
    CONF_DATA = {}

    def __init__(self, config_path='.command-engine.yml'):
        self.path = os.path.realpath(__file__)
        templates = glob.glob(os.path.join(os.path.dirname(self.path), 'templates', '*'))
        self.templates = {}
        for template in templates:
            (tpl_id, ext) = os.path.splitext(os.path.basename(template))
            self.templates[tpl_id] = open(template, 'r').read()

        with open(config_path, 'r') as handle:
            self.CONF_DATA = yaml.safe_load(handle)

        self.PROJECT_NAME = self.CONF_DATA['project_name']
        self.underlying_lib = import_module(self.CONF_DATA['module']['base_module'])
        self.IGNORE_LIST = self.CONF_DATA['module']['ignore']['funcs']
        # TODO: abstract
        func = getattr(self.underlying_lib, self.CONF_DATA['module']['instance_func'])
        self.obj = func(*self.CONF_DATA['module'].get('instance_args', []), **self.CONF_DATA['module'].get('instance_kwargs', {}))

    def template(self, template, opts):
        return self.templates[template] % opts

    @classmethod
    def __click_option(cls, name='arg', helpstr='TODO', ptype=None, default=None):
        args = [
            '"--%s"' % name,
            'help="%s"' % (helpstr.replace('"', '\\"') if helpstr else ""),
        ]
        if default:
            args.append('default="%s"' % default)
            args.append('show_default=True')
        if ptype is not None:
            args.extend(ptype)
        return '@click.option(\n%s\n)\n' % (',\n'.join(['    ' + x for x in args]))

    @classmethod
    def __galaxy_option(cls, name='arg', helpstr='TODO', ptype=None, default=None):
        return '\t' + cls.parameter_translation_galaxy(ptype).format(
            name=name,
            label=nice_name(name),
            help=('help="%s"' % helpstr.replace('"', '\\"') if helpstr else ""),
            default=default if default else 0
        ) + '\n'

    @classmethod
    def __click_argument(cls, name='arg', helpstr=None, ptype=None):
        args = [
            f'"{name}"'
        ]
        if ptype is not None:
            args.extend(ptype)

        # Multiple=True is not permitted for arguments. Because REASONS.

        if 'multiple=True' in args:
            args[0] = '"--' + args[0][1:]
            # Also help strings are not supported because they don't agree with
            # that philosophy and won't support it. (BOO.)
            #
            # But at least when it's coerced into a fake required option, we
            # can add help.
            #
            # If you try to make them all arguments it'll error on "didn't you
            # mean argument??"
            if helpstr is not None:
                args.append('help="%s"' % helpstr.replace('"', '\\"'))
            return f'@click.option({", ".join(args)}, required=True)\n'
        else:
            return f'@click.argument({", ".join(args)})\n'

    @classmethod
    def __galaxy_argument(cls, name='arg', ptype=None, desc=None):
        return '\t' + cls.parameter_translation_galaxy(ptype).format(
            name=name,
            label=nice_name(name),
            help='help="%s"' % desc,
            default=0,
        ) + '\n'

    @classmethod
    def load_module(cls, module_path):
        name = '.'.join(module_path)
        return import_module(name)

    def is_galaxyinstance(self, obj):
        # TODO: abstract
        return str(type(obj)) == self.CONF_DATA['module']['instance_cls']

    def is_function(self, obj):
        return str(type(obj)) == "<type 'instancemethod'>"

    def is_class(self, obj):
        return str(type(obj)).startswith('<class ')

    def recursive_attr_get(self, obj, section):
        if len(section) == 0:
            return obj
        elif len(section) == 1:
            try:
                return getattr(obj, section[0])
            except AttributeError:
                pass
        else:
            return getattr(self.recursive_attr_get(obj, section[0:-1]), section[-1])

    @classmethod
    def important_doc(cls, docstring):
        good = []
        if docstring is not None and len(docstring) > 0:
            for line in docstring.split('\n')[1:]:
                if line.strip() == '':
                    return ' '.join(good)
                else:
                    good.append(line.strip())
            return ' '.join(good)
        else:
            return "Warning: Undocumented Method"

    def flatten(self, x):
        # http://stackoverflow.com/a/577971
        result = []
        for el in x:
            if isinstance(el, list):
                if len(el) > 0:
                    if isinstance(el[0], str):
                        result.append(el)
                    else:
                        result.extend(self.flatten(el))
            else:
                result.append(el)
        return result

    @classmethod
    def _pt(cls, d, k, default=None):
        if 'typing.Union' in str(k):
            return default

        if 'bioblend.galaxy' in str(k):
            return d[dict]
        else:
            try:
                return d[k]
            except:
                raise Exception("Unknown parameter type " + k)

    @classmethod
    def parameter_translation(cls, k):
        return cls._pt(PARAM_TRANSLATION, k, default=[])

    @classmethod
    def parameter_translation_galaxy(cls, k):
        return cls._pt(PARAM_TRANSLATION_GALAXY, k, default="")

    @classmethod
    def parameter_translation_galaxy_cli(cls, k):
        return cls._pt(PARAM_TRANSLATION_GALAXY_CLI, k, default={'opt': "", 'arg': ""})

    def parse_arguments_defaults(self, func):
        argspec = inspect.getfullargspec(func)
        # Old
        # FullArgSpec(args=['self', 'b', 'c', 'd', 'e'], varargs=None, varkw=None,
        # defaults=(3, 'asdf'), kwonlyargs=[], kwonlydefaults=None,
        # annotations={})
        #
        # New
        # FullArgSpec(args=['self', 'b', 'c', 'd', 'e'], varargs=None, varkw=None,
        # defaults=(3, 'asdf'), kwonlyargs=[], kwonlydefaults=None,
        # annotations={'return': typing.List[dict], 'b': <class 'bool'>, 'c': typing.List[bool], 'd': <class 'int'>, 'e': <class 'str'>})

        # They need to be backwards to get defaults to match up right.
        args = argspec.args[::-1]

        # Remove self if present
        if args[-1] == 'self':
            args = args[:-1]

        if argspec.defaults is None:
            defaults = []
        else:
            defaults = argspec.defaults[::-1]

        # Convert all ``None`` to ""
        defaults = ["" if x is None else x for x in defaults]

        # Fill missing values with none, if more args than defaults.
        for i in range(len(args) - len(defaults)):
            defaults.append(None)

        args = OrderedDict(zip(args[::-1], defaults[::-1]))

        # Annotations
        # These are pulled from the docs
        annotations = self.parsedocs(func.__doc__)
        # But if we have better data, let's overwrite it with the corrections.
        for k, sigtype in argspec.annotations.items():
            doctype = annotations.get(k, {'type': None})['type']
            if sigtype != doctype:
                if sigtype not in (doctype, self.typing_translation(doctype)):
                    if 'typing' not in str(sigtype):
                        log.warning(f"Bioblend documentation doesn't match type signature: sig:{sigtype} doc:{doctype}")

                if k not in annotations:
                    annotations[k] = {}
                annotations[k]['type'] = sigtype

        return args, annotations

    def process(self, galaxy=False):
        for module in dir(self.obj):
            if module[0] == '_' or module[0].upper() == module[0]:
                continue
            # TODO: abstract.
            # chakin: ('debug', 'session', 'dbname', 'dbhost', 'dbport', 'dbuser', 'dbpass', 'dbschema', 'get_cvterm_id', 'get_cvterm_name')
            if module in self.CONF_DATA['module']['ignore']['top_attrs']:
                continue

            own_copy = getattr(self.obj, module)
            to_import = own_copy.__class__.__module__
            sm = import_module(to_import)

            submodules = dir(sm)
            # Find the "...Client"
            wanted = [x for x in submodules if 'Client' in x and x != 'Client'][0]
            self.process_client(module, sm, wanted, galaxy=galaxy)

    def process_client(self, module, sm, ssm_name, galaxy=False):
        log.info("Processing %s.%s", module, ssm_name)
        ssm = getattr(sm, ssm_name)
        for f in dir(ssm):
            if f[0] == '_' or f[0].upper() == f[0]:
                continue
            if f in self.IGNORE_LIST or '%s.%s' % (ssm, f) in self.IGNORE_LIST:
                continue

            func = getattr(ssm, f)
            if not callable(func):
                continue

            self.orig(module, sm, ssm, f, galaxy=galaxy)

        # Write module __init__
        with open(os.path.join(self.PROJECT_NAME, 'commands', self.CONF_DATA['module'].get('prefix', '') + module, '__init__.py'), 'w') as handle:
            pass

        with open(os.path.join(self.PROJECT_NAME, 'commands', 'cmd_%s%s.py' % (self.CONF_DATA['module'].get('prefix', ''), module)), 'w') as handle:
            handle.write('import click\n')
            # for function:
            files = list(glob.glob(self.PROJECT_NAME + "/commands/%s%s/*.py" % (self.CONF_DATA['module'].get('prefix', ''), module)))
            files = sorted([f for f in files if "__init__.py" not in f])
            for idx, path in enumerate(files):
                fn = path.replace('/', '.')[0:-3]
                fn_tail = path.split('/')[-1][0:-3]
                handle.write('from %s import cli as %s\n' % (fn, fn_tail))

            handle.write('\n\n@click.group()\n')
            handle.write('def cli():\n')
            if hasattr(ssm, "__doc__") and getattr(ssm, "__doc__"):
                handle.write('    """%s"""\n' % getattr(ssm, "__doc__"))
            handle.write('    pass\n\n\n')
            for i in files:
                fn_tail = i.split('/')[-1][0:-3]
                handle.write('cli.add_command(%s)\n' % fn_tail)

    def typing_translation(self, param_type):
        # If it's like 'str or list of str' just make it a 'list of str'
        if 'or list of' in str(param_type) and \
                (param_type.split(' ')[0] == param_type.split(' ')[-1] or \
                param_type.split(' ')[0] + 's' == param_type.split(' ')[-1]):
            param_type = 'list of ' + param_type.split(' ')[0]

        if 'or None' in str(param_type):
            param_type = param_type.split(' ')[0]

        # Coerce typing.Union[str, NoneType] to str
        if 'typing.Union[' in str(param_type) and 'NoneType]' in str(param_type) and str(param_type).count(',') == 1:
            param_type = str(param_type)[len('typing.Union['):str(param_type).index(',')]
            if 'typing' in param_type: # (if it's a Union[List[str], None])
                param_type = eval(param_type) # Ewww

        if param_type == 'bool':
            return bool
        elif param_type == 'int':
            return int
        elif param_type == 'float':
            return float
        elif param_type == 'dict':
            return dict
        elif param_type == 'str':
            return str
        elif param_type == 'list':
            return list
        elif param_type in ('list of dicts', 'list of dict'):
            return List[dict]
        elif param_type in ('list of strings', 'list of string', 'list of str'):
            return List[str]
        elif param_type in ('list of int', 'list of ints'):
            return List[int]
        elif param_type in ('list of bool', 'list of bools'):
            return List[bool]
        elif param_type in ('list of float', 'list of floats'):
            return List[float]
        elif 'typing' in str(param_type):
            return param_type
        elif inspect.isclass(param_type):
            return param_type
        elif param_type == 'object':  # Gross but not my fault.
            return dict
        elif param_type is None or param_type == 'None' or param_type == 'NoneType':
            return None
        elif param_type == 'file':
            return 'file'
        elif 'bioblend.' in param_type:
            return dict
        print("HIIIIIIII an unsupported param type", param_type)

    def parsedocs(self, argdoc):
        param_docs = {}

        sections = [x for x in argdoc.split("\n\n")]
        sections = [re.sub('\s+', ' ', x.strip()) for x in sections if x != '']
        paramre = re.compile(":type (?P<param_name>[^:]+): (?P<param_type>[^:]+) :param (?P<param_name2>[^:]+):\s*(?P<desc>.*)")
        returnre = re.compile(":rtype:\s*(?P<param_type>[^:]+)\s*(?P<ret>:returns?:)\s*(?P<desc>.*)$")
        rereturn = re.compile("(?P<ret>:returns?:)\s*(?P<desc>.+)\s*:rtype:\s*(?P<param_type>[^:]+)")
        for subsec in sections:
            m = paramre.match(subsec)
            print(subsec, m)
            if m:
                if self.CONF_DATA.get('strict', False) == True:
                    assert m.group('param_name') == m.group('param_name2')

                param_type = m.group('param_type').strip()
                guessed_type = self.typing_translation(param_type)
                if guessed_type is not None:
                    param_type = guessed_type

                param_docs[m.group('param_name')] = {
                    'type': param_type,
                    'desc': m.group('desc').strip()
                }
            m = returnre.match(subsec)

            # If first regex fails, try second
            if not m:
                m = rereturn.match(subsec)

            if m:
                param_type = m.group('param_type').strip()
                guessed_type = self.typing_translation(param_type)
                if guessed_type is not None:
                    param_type = guessed_type
                param_docs['return'] = {
                    'type': param_type,
                    'desc': argdoc[argdoc.index(m.group('ret')) + len(m.group('ret')):].strip(),
                }
        return param_docs

    def orig(self, module_name, submodule, subsubmodule, function_name, galaxy=False):
        target = [module_name, function_name]
        log.debug("Building %s", '.'.join(target))

        func = getattr(subsubmodule, function_name)
        candidate = '.'.join(target)

        argdoc = func.__doc__

        data = {
            'project_name': self.PROJECT_NAME,
            'meta_module_name': module_name,
            'meta_function_name': function_name,
            'command_name': function_name,
            'click_arguments': "",
            'click_options': "",
            'args_with_defaults': "ctx",
            'wrapped_method_args': "",
            # Galaxy stuff
            'galaxy_arguments': "    <!-- arguments -->\n",
            'galaxy_options': "    <!-- options -->\n",
            'galaxy_cli_arguments': "",
            'galaxy_cli_options': "",
            # By default we output JSON, so we sort the keys for
            # reproducibility. however in some cases we don't want that,
            # we'll want text outputs.
            'galaxy_reformat_json': '| jq -S .',
            'galaxy_output_format': 'json',
        }


        deprecated = False
        argspec, annotations = self.parse_arguments_defaults(func)

        data['kwarg_updates'] = ''
        data['empty_kwargs'] = ''

        __import__('pprint').pprint(argspec)
        __import__('pprint').pprint(annotations)
        # Ignore with only cls/self
        if len(argspec) > 0:
            method_signature = ['ctx']
            # Args and kwargs are separate, as args should come before kwargs
            method_signature_args = []
            method_signature_kwargs = []
            method_exec_args = []
            method_exec_kwargs = []

            def process_arg(k, v, param_type, real_type):
                log.debug("Processing %s=%s %s %s", k, v, param_type, real_type)
                orig_v = copy.deepcopy(v)
                # If v is not None, then it's a kwargs, otherwise an arg
                if v is not None:
                    # Strings must be treated specially by removing their value
                    if v == '__None__':
                        v = 'None'
                        orig_v = None
                    elif isinstance(v, str):
                        v = '"%s"' % v

                    if v == []:
                        v = None
                        orig_v = None
                    # All other instances of V are fine, e.g. boolean=False or int=1000

                    # Register twice as the method invocation uses v=k
                    if v != 'None':
                        method_signature_kwargs.append("%s=%s" % (k, v))
                        if real_type == 'dict':
                            v = 'json_loads(%s)' % v
                        method_exec_kwargs.append('%s=%s' % (k, k))
                    else:
                        # Add to signature, but NOT exec because we take care of that elsewhere.
                        method_signature_kwargs.append("%s=%s" % (k, v))

                    # TODO: refactor
                    try:
                        descstr = annotations[k]['desc']
                    except KeyError:
                        log.warning("Error finding %s in %s" % (k, candidate))
                        descstr = None
                    data['click_options'] += self.__click_option(name=k, helpstr=descstr, ptype=param_type, default=orig_v)

                    data['galaxy_options'] += self.__galaxy_option(name=k, helpstr=descstr, ptype=real_type, default=orig_v)
                    data['galaxy_cli_options'] += self.parameter_translation_galaxy_cli(real_type)['opt'].format(name=k) + '\n'
                else:
                    # Args, not kwargs
                    method_signature_args.append(k)
                    if real_type == 'dict':
                        tk = 'json_loads(%s)' % k
                    else:
                        tk = k
                    method_exec_args.append(tk)
                    try:
                        descstr = annotations[k]['desc']
                    except KeyError:
                        log.warning("Error finding %s in %s" % (k, candidate))
                        descstr = None
                    data['click_arguments'] += self.__click_argument(name=k, ptype=param_type, helpstr=descstr)
                    data['galaxy_arguments'] += self.__galaxy_argument(name=k, ptype=real_type, desc=descstr)
                    data['galaxy_cli_arguments'] += self.parameter_translation_galaxy_cli(real_type)['arg'].format(name=k) + '\n'

            argspec_keys = argspec.keys()
            for k, v in argspec.items():
                # TODO: replace __return__ with return.
                if k == 'return':
                    continue

                real_type = self.typing_translation(annotations[k]['type'])
                param_type = self.parameter_translation(real_type)

                process_arg(k, v, param_type, real_type)

            had_weird_kwargs = False
            for k in sorted(annotations.keys()):
                if k == 'return':
                    continue
                # Ignore things we've seen before
                if k in argspec_keys:
                    continue
                real_type = self.typing_translation(annotations[k]['type'])
                param_type = self.parameter_translation(real_type)
                if 'list' in str(real_type) or 'List' in str(real_type):
                    default_value = []
                else:
                    default_value = '__None__'

                process_arg(k, default_value, param_type, real_type)
                # Booleans are diff
                if param_type == 'bool':
                    data['kwarg_updates'] += "    if %s is not None:\n        kwargs['%s'] = %s\n" % (k, k, k)
                elif param_type == 'str':
                    data['kwarg_updates'] += "    if %s and len(%s) > 0:\n        kwargs['%s'] = %s\n" % (k, k, k, k)
                had_weird_kwargs = True

            # Complete args
            data['args_with_defaults'] = ', '.join(method_signature +
                                                   method_signature_args +
                                                   method_signature_kwargs)
            data['wrapped_method_args'] = ', '.join(method_exec_args +
                                                    method_exec_kwargs)
            if had_weird_kwargs:
                data['wrapped_method_args'] += ', **kwargs'
                data['empty_kwargs'] = '\n    kwargs = {}\n'


        # TODO: rtype -> dict_output / list_output / text_output
        # return must be in param_docs or it's a documentation BUG.
        if 'return' not in annotations:
            if '.. deprecated::' in argdoc:
                deprecated = True

            if self.CONF_DATA.get('strict', False) == True:
                if not deprecated:
                    raise Exception("%s is not documented with a return type" % candidate)
            else:
                annotations['return'] = {
                    'type': dict,
                    'desc': '',
                }
                if not deprecated:
                    log.warning("%s is not documented with a return type" % candidate)

        data['output_format'] = annotations['return']['type']
        if data['output_format'] == 'None':
            # Usually means writing to stdout.
            data['galaxy_reformat_json'] = ''
            data['galaxy_output_format'] = 'txt'

        # All of them behave basically the same.
        # Either it's structured data and we turn it into json
        # or it's a simple value and we turn it into text.
        if 'Union' in str(data['output_format']):
            raise Exception("UNION outputs are unsupported")
        elif 'List[' in str(data['output_format']):
            data['output_format'] = 'json'
        elif data['output_format'] in (list, dict, tuple):
            data['output_format'] = 'json'
        else:
            data['output_format'] = 'text'

        data['output_documentation'] = annotations['return'].get('desc', "").strip()

        # My function is more effective until can figure out docstring
        data['short_docstring'] = self.important_doc(argdoc)
        # Full method call
        data['wrapped_method'] = f"ctx.{self.CONF_DATA['module']['instance_ctx']}.{candidate}"

        # Generate a command name, prefix everything with auto_ to identify the
        # automatically generated stuff
        cmd_name = '%s.py' % function_name
        cmd_path = os.path.join(self.PROJECT_NAME, 'commands', self.CONF_DATA['module'].get('prefix', '') + module_name, cmd_name)
        if not os.path.exists(os.path.join(self.PROJECT_NAME, 'commands', self.CONF_DATA['module'].get('prefix', '') + module_name)):
            os.makedirs(os.path.join(self.PROJECT_NAME, 'commands', self.CONF_DATA['module'].get('prefix', '') + module_name))

        # Save file
        if deprecated:
            if os.path.exists(cmd_path):
                os.unlink(cmd_path)
        else:
            with open(cmd_path, 'w') as handle:
                handle.write(self.template('click', data))

        if galaxy:
            tool_name = '%s_%s.xml' % (module_name, function_name)
            if not os.path.exists('galaxy'):
                os.makedirs('galaxy')
            tool_path = os.path.join('galaxy', tool_name)
            with open(tool_path, 'w') as handle:
                handle.write(self.template('galaxy', data))



def fn(self, b, c, d=3, e="asdf", q="bsdf"):
    """
    Get the list of all installed datatypes.

    :type b: bool
    :param b: Return only the extension rather than the datatype name

    :type c: list of bool
    :param c: Whether to return only datatypes which can be uploaded

    :type d: int
    :param d: Some int

    :type e: str
    :param e: Some text

    :type q: str
    :param q: Some text

    :rtype: list of dict
    :return: A list of dicts
        For example asdf
    """
    return 1

def fn2(self, b: bool, c: List[bool], d:int=3 , e:str="asdf", q="bsdf") -> List[dict]:
    """
    Get the list of all installed datatypes.

    :type b: bool
    :param b: Return only the extension rather than the datatype name

    :type c: list of bool
    :param c: Whether to return only datatypes which can be uploaded

    :type d: int
    :param d: Some int

    :type e: str
    :param e: Some text

    :type q: str
    :param q: Some text

    :rtype: list of dict
    :return: A list of dicts
        For example asdf
    """
    return 1

z = ScriptBuilder()

# print('fn(self, b, c, d=3, e="asdf")')
# __import__('pprint').pprint(z.parse_arguments_defaults(fn))
# print('fn2(self, b: bool, c: List[bool], d=3, e="asdf") -> List[dict]')
# __import__('pprint').pprint(z.parse_arguments_defaults(fn2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process libraries into CLI tools')
    parser.add_argument('--galaxy', action='store_true', help="Write out galaxy tools as well")
    parser.add_argument('--config', help="Path to command-engine.yml file", default='.command-engine.yml')
    args = parser.parse_args()
    z = ScriptBuilder(config_path=args.config)
    z.process(galaxy=args.galaxy)

