#!/usr/bin/env python
import os
import sys
from string import Template
import yaml
from importlib import import_module
from click.testing import CliRunner

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_dir)
with open('.command-engine.yml', 'r') as handle:
    CONF_DATA = yaml.safe_load(handle)

# PY3K ONLY for the import.
if 'docs_import' in CONF_DATA:
    for module in CONF_DATA['docs_import']:
        vars()[module] = import_module(module)

cli_module = import_module(CONF_DATA['project_name'] + '.cli')

base_cli = getattr(cli_module, CONF_DATA['project_name'])
runner = CliRunner()

COMMAND_TEMPLATE = Template('''
``${subcommand}`` command
${module_underline}

${command_help}
''')

COMMANDS_TEMPLATE = """========
Commands
========

%s

.. toctree::
   :maxdepth: 0
""" % CONF_DATA['documentation']

command_doc_dir = os.path.join("docs", "commands")
commands = COMMANDS_TEMPLATE

for command in cli_module.list_cmds():
    if command == 'init':
        # Skip documenting init because it's special
        continue

    commands += "\n   commands/%s.rst" % command
    parent_doc_handle = open(os.path.join(command_doc_dir, command + ".rst"), "w")
    parent_doc_handle.write('%s\n' % command)
    parent_doc_handle.write('%s\n' % ('=' * len(command)))
    parent_doc_handle.write(Template("""
This section is auto-generated from the help text for the ${library} command
``${command}``.

""").safe_substitute(command=command, library=CONF_DATA['project_name']))

    for subcommand in cli_module.list_subcmds(command):
        if 'docs_reset_hook' in CONF_DATA:
            eval(CONF_DATA['docs_reset_hook'])

        command_obj = cli_module.name_to_command(command, subcommand)

        function = command_obj.callback
        raw_rst = function.__doc__

        def clean_rst_line(line, remove=4):
            if line.startswith(" " * remove):
                return line[remove:]
            else:
                return line
        clean_rst = "\n".join(map(clean_rst_line, raw_rst.split("\n")))
        if 'Output:' in clean_rst:
            output_rst = clean_rst[clean_rst.index('Output:') + len('Output:'):].lstrip('\n')
            clean_rst = clean_rst[0:clean_rst.index('Output:')]
            output_rst = "\n".join([clean_rst_line(x, remove=5) for x in raw_rst.split("\n")])
        else:
            output_rst = ""

        result = runner.invoke(base_cli, [command, subcommand, "--help"])
        print(subcommand, result)
        output = result.output
        lines = output.split("\n")
        new_lines = []
        help_lines = False
        option_lines = False
        output_lines = False

        for line in lines:
            if line.startswith("Usage: "):
                new_lines.append("**Usage**::\n\n    %s" % line[len("Usage: "):])
                new_lines.append("\n**Help**\n")
                new_lines.append(clean_rst)
                help_lines = True
                option_lines = False
                output_lines = False
            elif line.startswith("Options:"):
                help_lines = False
                option_lines = True
                output_lines = False
                new_lines.append("**Options**::\n\n")
            elif line.strip().startswith("Output:"):
                help_lines = False
                option_lines = False
                output_lines = True
                new_lines.append("**Output**\n\n")
                new_lines.append(output_rst[output_rst.index('Output:') + len('Output:'):].lstrip('\n'))
            elif option_lines:
                new_lines.append("    %s" % line)
            elif output_lines:
                pass
        text = COMMAND_TEMPLATE.safe_substitute(
            command=command,
            subcommand=subcommand,
            command_help="\n".join(new_lines),
            module_underline="-" * (len(subcommand) + len('```` command'))
        )
        parent_doc_handle.write(text)

open(os.path.join("docs", "commands.rst"), "w").write(commands)
