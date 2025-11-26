from os.path import dirname, basename, isfile, join
import glob
import importlib

# 自动发现并导入此目录下的所有模块，以触发其中的注册函数
module_names = []
for f in glob.glob(join(dirname(__file__), "*.py")):
    if isfile(f) and not f.endswith('__init__.py'):
        module_names.append(basename(f)[:-3])

__all__ = module_names

for module_name in __all__:
    importlib.import_module(f'.{module_name}', __package__)
